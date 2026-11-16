"""本机仿真专用 DDS 后端：真实 LowCmd 传输 + 自定义全状态反馈。"""
import argparse
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass

import mujoco
import numpy as np
from cyclonedds.idl import IdlStruct
from cyclonedds.idl.types import array, float64, uint32
from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_

DOMAIN = 73
CMD_TOPIC = 'rt/lowcmd'
STATE_TOPIC = 'rt/holomotion_sim_state'

@dataclass
class SimState(IdlStruct, typename='holomotion_sim::SimState'):
    seq: uint32
    qpos: array[float64, 36]
    qvel: array[float64, 35]

def initialize():
    # 固定回环网卡、独立域、单播发现；没有实体机器人网卡/域选择入口。
    import unitree_sdk2py.core.channel as channel
    channel.ChannelConfigHasInterface='''<CycloneDDS><Domain Id="any"><General><Interfaces><NetworkInterface name="$__IF_NAME__$"/></Interfaces><AllowMulticast>false</AllowMulticast></General><Discovery><ParticipantIndex>auto</ParticipantIndex><Peers><Peer Address="127.0.0.1"/></Peers></Discovery></Domain></CycloneDDS>'''
    ChannelFactoryInitialize(DOMAIN, 'lo')

def inbox():
    messages = queue.Queue(maxsize=2)
    def receive(msg):
        try:
            messages.put_nowait(msg)
        except queue.Full:
            try: messages.get_nowait()
            except queue.Empty: pass
            messages.put_nowait(msg)
    return messages, receive

class DDSBackend:
    """控制端不 mj_step；每个物理子步通过 DDS 交给独立进程推进。"""
    def __init__(self, ev):
        self.ev = ev
        self.child = None
        self.seq = 0
        self.pairs = 0
        self.temp = tempfile.TemporaryDirectory(prefix='holomotion-dds-')
        self.folder = Path(self.temp.name)
        session=self.folder.name.replace('-', '_')
        self.cmd_topic=f'rt/holomotion_sim/{session}/lowcmd'
        self.state_topic=f'rt/holomotion_sim/{session}/state'
        self.rx = self.tx = None
        if (ev.m.nq, ev.m.nv, ev.m.nu) != (36, 35, 29):
            raise ValueError('DDS 后端仅支持当前 29 关节浮动根部 G1')
        # 用同一 XML 与名称映射，明确采用本地执行器顺序而非假定 SDK 电机顺序。
        self.joints = [ev.m.joint(int(ev.m.actuator_trnid[i, 0])).name for i in range(ev.m.nu)]
        self.mu = [ev.mjcf_dof_names.index(name) for name in self.joints]
        self.write_initial()
        try:
            initialize()
            self.messages, receive = inbox()
            self.rx = ChannelSubscriber(self.state_topic, SimState); self.rx.Init(receive, 1)
            self.tx = ChannelPublisher(self.cmd_topic, LowCmd_); self.tx.Init()
            config = dict(model=str(Path(ev.config.robot_xml_path).resolve()), timestep=float(ev.m.opt.timestep), joints=self.joints, cmd_topic=self.cmd_topic, state_topic=self.state_topic)
            (self.folder/'config.json').write_text(json.dumps(config))
            self.child = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), '--worker', str(self.folder), '--parent', str(os.getpid())])
            self.wait_for(0, 20)
            # 显式握手确认控制方向的发现完成；不推进物理时间。
            self.reset()
        except Exception:
            self.close()
            raise

    def write_initial(self):
        np.savez(self.folder/'initial.npz', qpos=self.ev.d.qpos, qvel=self.ev.d.qvel)

    def wait_for(self, seq, timeout=2):
        deadline = time.monotonic()+timeout
        while time.monotonic()<deadline:
            if self.child is not None and self.child.poll() is not None:
                raise RuntimeError('DDS 仿真进程已退出')
            try: state=self.messages.get(timeout=min(.2, max(.001,deadline-time.monotonic())))
            except queue.Empty: continue
            if state.seq == seq:
                if not np.isfinite(state.qpos).all() or not np.isfinite(state.qvel).all():
                    raise RuntimeError('DDS 反馈包含非有限数值')
                return state
        raise RuntimeError('DDS 指令/状态超时，仿真已停止')

    def reset(self):
        self.write_initial()
        # reserve[0] 是本原型的锁步序号，reserve[1] 是仿真复位命令。
        self.seq += 1
        cmd=unitree_hg_msg_dds__LowCmd_(); cmd.reserve[0]=self.seq; cmd.reserve[1]=1
        # 初始化发现期间重发相同复位序号是幂等的。
        for _ in range(10):
            self.tx.Write(cmd)
            try: self.wait_for(self.seq, .5); return
            except RuntimeError:
                if self.child.poll() is not None: raise
        raise RuntimeError('DDS 复位握手超时')

    def step(self, sleep=False):
        ev=self.ev
        for _ in range(ev.control_decimation):
            self.seq += 1
            cmd=unitree_hg_msg_dds__LowCmd_(); cmd.reserve[0]=self.seq
            for i, (name, mu) in enumerate(zip(self.joints, self.mu)):
                c=cmd.motor_cmd[i]; c.mode=1
                c.q=float(ev.target_dof_pos_by_name[name]); c.dq=0.; c.tau=0.
                c.kp=float(ev.kps_mu[mu]); c.kd=float(ev.kds_mu[mu])
            self.tx.Write(cmd)
            state=self.wait_for(self.seq)
            # 本地仅镜像仿真状态供观测与渲染，不在控制进程重复推进物理。
            ev.d.qpos[:]=state.qpos; ev.d.qvel[:]=state.qvel
            mujoco.mj_forward(ev.m, ev.d)
            self.pairs += 1

    def close(self):
        if self.child is not None and self.child.poll() is None:
            self.child.terminate()
            try: self.child.wait(timeout=5)
            except subprocess.TimeoutExpired: self.child.kill(); self.child.wait()
        if self.rx is not None: self.rx.Close()
        if self.tx is not None: self.tx.Close()
        self.temp.cleanup()

def worker(folder, parent):
    initialize()
    folder=Path(folder); cfg=json.loads((folder/'config.json').read_text())
    m=mujoco.MjModel.from_xml_path(cfg['model']); m.opt.timestep=cfg['timestep']; d=mujoco.MjData(m)
    names=[m.joint(int(m.actuator_trnid[i,0])).name for i in range(m.nu)]
    if names != cfg['joints']: raise RuntimeError('DDS 两端执行器名称顺序不一致')
    def reset():
        initial=np.load(folder/'initial.npz')
        mujoco.mj_resetData(m,d); d.qpos[:]=initial['qpos']; d.qvel[:]=initial['qvel']; mujoco.mj_forward(m,d)
    reset()
    messages,receive=inbox()
    rx=ChannelSubscriber(cfg['cmd_topic'],LowCmd_); rx.Init(receive,1)
    tx=ChannelPublisher(cfg['state_topic'],SimState); tx.Init()
    last_seq=0; last_command=time.monotonic()
    while os.getppid()==parent:
        try: cmd=messages.get(timeout=.2)
        except queue.Empty:
            if last_seq==0: tx.Write(SimState(0,d.qpos.tolist(),d.qvel.tolist()))
            # 无命令时冻结物理时间，匹配网页暂停/丢帧语义。
            continue
        seq=int(cmd.reserve[0]); last_command=time.monotonic()
        if seq < last_seq: continue
        if seq > last_seq:
            if cmd.reserve[1]: reset()
            else:
                for i in range(m.nu):
                    j=int(m.actuator_trnid[i,0]); qa=int(m.jnt_qposadr[j]); va=int(m.jnt_dofadr[j]); c=cmd.motor_cmd[i]
                    values=[c.q,c.dq,c.tau,c.kp,c.kd]
                    if not np.isfinite(values).all() or c.kp<0 or c.kd<0: raise RuntimeError('无效 DDS 电机指令')
                    tau=c.tau+c.kp*(c.q-d.qpos[qa])+c.kd*(c.dq-d.qvel[va])
                    d.ctrl[i]=np.clip(tau,*m.actuator_ctrlrange[i]) if m.actuator_ctrllimited[i] else tau
                mujoco.mj_step(m,d)
            last_seq=seq
        tx.Write(SimState(seq,d.qpos.tolist(),d.qvel.tolist()))
    rx.Close(); tx.Close()

if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--worker',required=True); parser.add_argument('--parent',type=int,required=True)
    args=parser.parse_args()
    try: worker(args.worker,args.parent)
    finally: os._exit(0)
