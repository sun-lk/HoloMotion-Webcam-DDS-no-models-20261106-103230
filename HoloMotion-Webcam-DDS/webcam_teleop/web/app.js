// 页面主线程：显示摄像头与骨架、响应按钮、收发 HTTP；识别在 Worker 中运行。
const $=id=>document.getElementById(id);
const video=$('video'),canvas=$('overlay'),ctx=canvas.getContext('2d');
// 骨架连线使用 MediaPipe 的关键点编号，用于画图，不是机器人的关节编号。
const edges=[[11,12],[11,13],[13,15],[12,14],[14,16],[11,23],[12,24],[23,24],[23,25],[25,27],[24,26],[26,28],[27,29],[29,31],[28,30],[30,32]];
// busy 防识别任务堆积；serverBusy 限制同时只有一个关键点上传请求。
let worker,stream,busy=false,ready=false,sequence=0,lastImageTime=-1,timer,poses=0,lastRate=performance.now(),latestStatus={},serverBusy=false;
// 摄像头选择：记住上次设备；deviceId 失效（拔掉/被占用）时自动回退默认摄像头。
const cameraSelect=$('cameraSelect');
for(const [id,port] of [['simEntry','8766'],['realEntry','8765']]){
  const url=new URL(location.href);url.port=port;url.pathname='/';url.search='';url.hash='';$(id).href=url.href;
}
let cameraId=localStorage.getItem('holomotion.cameraId')||'';
function notice(text){$('notice').textContent=text;}
// 控制协议示例：POST /api/control，JSON 为 {action:"calibrate"}。
async function command(action,extra={}){const response=await fetch('/api/control',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({action,...extra})});const data=await response.json();if(!response.ok)throw Error(data.error);return data;}
// 创建独立识别线程，先等 ready，再接收 pose 或 error 消息。
async function initWorker(){
  if(ready)return;
  worker=new Worker('/pose-worker.js');
  await new Promise((resolve,reject)=>{
    worker.onmessage=({data})=>{if(data.type==='ready'){ready=true;resolve();}else if(data.type==='error')reject(Error(data.message));};
    worker.onerror=e=>reject(Error(e.message)); worker.postMessage({type:'init'});
  });
  worker.onmessage=async({data})=>{
    if(data.type==='error'){busy=false;notice('人体识别失败：'+data.message);return;}
    if(data.type!=='pose')return;
    busy=false;ctx.clearRect(0,0,canvas.width,canvas.height);
    if(data.landmarks&&data.world){
      // landmarks 是归一化图像坐标，仅用来把骨架画到视频画布上。
      const points=data.landmarks;
      ctx.strokeStyle='#62f2c2';ctx.fillStyle='#ffe5a0';ctx.lineWidth=3;
      for(const[a,b]of edges){if(Math.min(points[a].visibility??0,points[b].visibility??0)<.5)continue;ctx.beginPath();ctx.moveTo(points[a].x*canvas.width,points[a].y*canvas.height);ctx.lineTo(points[b].x*canvas.width,points[b].y*canvas.height);ctx.stroke();}
      for(const p of points){if((p.visibility??0)<.5)continue;ctx.beginPath();ctx.arc(p.x*canvas.width,p.y*canvas.height,4,0,Math.PI*2);ctx.fill();}
      const confidence=Math.min(...[11,12,13,14,15,16,23,24].map(i=>data.world[i].visibility??0));
      $('poseState').textContent=confidence>=.5?'已识别人形 · 上半身可见':'请让肩、肘、手腕、髋部完整入镜';
      // world 是人体三维世界关键点；只发送 33×4 数字，不发送视频图像。
      // sequence 递增，用来让后端拒绝乱序帧。visibility 表示可见度。
      if(!serverBusy){serverBusy=true;try{const response=await fetch('/api/pose',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({sequence:sequence++,timestamp_ms:data.timestamp,landmarks:data.world.map(p=>[p.x,p.y,p.z,p.visibility??0])})});if(!response.ok){const r=await response.json();notice(r.error);}}catch(e){notice('关键点发送失败：'+e.message);}finally{serverBusy=false;}}
    }else{$('poseState').textContent='未检测到人体，请后退并完整入镜';}
    poses++;const now=performance.now();if(now-lastRate>1000){$('sourceFps').textContent=`识别 ${(poses*1000/(now-lastRate)).toFixed(1)} Hz`;poses=0;lastRate=now;}
  };
}
// 分辨率/帧率保持与原逻辑一致；选了具体设备才加 deviceId 约束。
function videoConstraints(){
  const fast=location.port==='8766';
  const base={width:{ideal:640},height:{ideal:480},frameRate:{ideal:fast?30:24}};
  return cameraId?{...base,deviceId:{exact:cameraId}}:base;
}
// 设备名要在授权之后才可读，所以先开流再枚举；顺便把当前实际设备设为选中项。
async function listCameras(){
  try{
    const devices=await navigator.mediaDevices.enumerateDevices();
    const cams=devices.filter(d=>d.kind==='videoinput');
    cameraSelect.innerHTML=cams.length?cams.map((d,i)=>`<option value="${d.deviceId}">${d.label||`摄像头 ${i+1}`}</option>`).join(''):'<option value="">未检测到摄像头</option>';
    const active=stream?.getVideoTracks()[0]?.getSettings?.().deviceId;
    if(active&&cams.some(c=>c.deviceId===active))cameraSelect.value=active;
  }catch(e){notice('无法列出摄像头：'+e.message);}
}
// 按当前选择开流；指定设备不可用时清掉记忆并回退默认，避免插拔后打不开。
async function openStream(){
  try{stream=await navigator.mediaDevices.getUserMedia({video:videoConstraints(),audio:false});}
  catch(e){
    if(!cameraId)throw e;
    cameraId='';localStorage.removeItem('holomotion.cameraId');notice('所选摄像头不可用，已回退默认摄像头：'+e.message);
    stream=await navigator.mediaDevices.getUserMedia({video:videoConstraints(),audio:false});
  }
  video.srcObject=stream;await video.play();canvas.width=video.videoWidth;canvas.height=video.videoHeight;
}
// 只有用户点击按钮才申请摄像头；audio:false 表示不采集麦克风。
async function startCamera(){
  $('camera').disabled=true;notice('正在加载本地人体识别模型…');
  try{
    await initWorker();
    await openStream();await listCameras();
    await command('new_camera');sequence=0;lastImageTime=-1;
    // 50 ms 检查一次，识别提交上限约 20 Hz；摄像头/机器性能会影响实际频率。
    // ImageBitmap 通过 transfer 移交 Worker，避免在主线程执行识别。
    clearInterval(timer);timer=setInterval(async()=>{if(busy||video.readyState<2||video.currentTime===lastImageTime)return;busy=true;lastImageTime=video.currentTime;try{const bitmap=await createImageBitmap(video);worker.postMessage({type:'frame',bitmap,timestamp:performance.now()},[bitmap]);}catch(e){busy=false;notice(e.message);}},location.port==='8766'?16:50);
    notice('请正对镜头自然站立，双臂垂下，然后点击“校准站姿”。');
  }catch(e){notice('摄像头启动失败：'+e.message);stream?.getTracks().forEach(t=>t.stop());stream=null;$('camera').disabled=false;}
}
$('camera').onclick=startCamera;
$('displaySource').onchange=async()=>{
  try{await command('set_display_source',{source:$('displaySource').value});}
  catch(e){$('displaySource').value=latestStatus.display_source||'simulation';notice(e.message);}
};
$('feedbackSource').onchange=async()=>{
  try{await command('set_feedback_source',{source:$('feedbackSource').value});notice('策略反馈已切换，请重新采集并对齐机器人后开始跟随');}
  catch(e){$('feedbackSource').value=latestStatus.feedback_source||'robot';notice(e.message);}
};
// 真机开关：关闭时只跑本地 MuJoCo 仿真，策略照常推理但不下发任何命令。
$('robotToggle').onclick=async()=>{
  const next=!latestStatus.robot_enabled;
  try{await command('set_robot',{enabled:next});notice(next?'已打开真机开关：策略输出会下发到机器人':'已关闭真机开关：只跑本地仿真，不再下发命令');}
  catch(e){notice(e.message);}
};
// 切换设备：已开流就立刻重开并重置后端序列（此时必须先重新校准）；未开流只记住选择。
cameraSelect.onchange=async()=>{
  cameraId=cameraSelect.value;localStorage.setItem('holomotion.cameraId',cameraId);
  if(!stream)return;
  try{stream.getTracks().forEach(t=>t.stop());await openStream();await command('new_camera');sequence=0;lastImageTime=-1;notice('已切换摄像头，请重新校准站姿。');}
  catch(e){notice('切换摄像头失败：'+e.message);}
};
// 插拔摄像头只刷新列表，不打断当前采集。
navigator.mediaDevices?.addEventListener?.('devicechange',()=>{if(stream)listCameras();});
for(const[id,action]of [['calibrate','calibrate'],['follow','start'],['pause','pause'],['reset','reset']])$(id).onclick=async()=>{try{const actual=action==='calibrate'&&latestStatus.backend==='g1ra'?'align':action;await command(actual,actual==='align'?{move_robot:true}:{});notice(actual==='align'?'已冻结人体目标，将缓慢移动实机肩肘；可随时点击暂停停止':actual==='calibrate'?'正在校准…':'');}catch(e){notice(e.message);}};
// 关闭摄像头时停止媒体轨道，并向后端发 pause。
$('close').onclick=async()=>{clearInterval(timer);stream?.getTracks().forEach(t=>t.stop());stream=null;video.srcObject=null;ctx.clearRect(0,0,canvas.width,canvas.height);$('camera').disabled=false;$('poseState').textContent='摄像头已关闭';await command('pause').catch(()=>{});};
// 每 500 ms 查询状态，控制按钮是否可用；与关键点上传独立。
// 发送频率用相邻两次轮询的增量估算，暂停时会自然归零。
let lastSent=0,lastSentAt=performance.now();
// 与 g1-remote-action 的状态机对齐，便于判断机器人当前能不能接动作。
const robotStates={0:'未知',1:'零力矩',2:'回默认位',3:'策略跟随',4:'策略保持',5:'急停'};
const diagnostics=document.createElement('div');
diagnostics.style.cssText='color:#62f2c2;padding:10px 0;font-size:16px';
$('stage').after(diagnostics);
async function poll(){try{const response=await fetch('/api/status');latestStatus=await response.json();
if(location.port==='8766'){
 const s=latestStatus, low=(s.pose_low_visibility||[]).join('、')||'无';
 diagnostics.textContent=`青色细骨架=映射目标；实体模型=物理跟随。低可见度：${low}；近1秒低可见度帧 ${s.pose_low_visibility_percent??'—'}%（${s.pose_input_window_frames??0}帧）；人体数据 ${s.pose_age_ms??'—'}ms；映射 ${s.mapping_ms??'—'}ms；推理物理 ${s.inference_ms??'—'}ms；渲染 ${s.render_ms??'—'}ms [${s.render_device??'设备检测中'}]；最大关节参考误差 ${s.reference_tracking_error_deg??'—'}°`;
}
robotStates[3]=latestStatus.g1ra_control_mode==='arm_sdk'?'手臂跟随':'全身 POLICY';
robotStates[4]=latestStatus.g1ra_control_mode==='arm_sdk'?'手臂保持':'全身 POLICY_HOLD';
// 真机模式显示两条链路的连接状态、发送节奏和回传年龄；绿色代表状态链路通。
const now=performance.now(),sent=latestStatus.action_sent??0;
const sentHz=(now-lastSentAt)>200?(sent-lastSent)/((now-lastSentAt)/1000):0;
lastSent=sent;lastSentAt=now;$('transport').style.color=latestStatus.backend==='g1ra'?(latestStatus.link_state?'#77dfbf':'#ff8b8b'):'';
$('transport').textContent=latestStatus.backend==='dds'?`DDS 仿真 · ${latestStatus.dds_pairs??0} 次反馈`:(latestStatus.backend==='g1ra'?`真机 G1 · 动作${latestStatus.link_action?'已连':'未连'} · 状态${latestStatus.link_state?'已连':'未连'} · 发送 ${sentHz>0.05?sentHz.toFixed(1)+' Hz':'—'} · 已发 ${sent} 帧 · 回传 ${latestStatus.state_age_ms==null?'—':latestStatus.state_age_ms.toFixed(0)+' ms'} · 机器人 ${robotStates[latestStatus.robot_state]??latestStatus.robot_state??'—'} · 实机位姿接口 ${latestStatus.pose_link?(latestStatus.pose_link.ok?'在线':'离线'):'未配置'}`:'直接仿真');$('stage').textContent=latestStatus.stage;$('policyFps').textContent=`控制 ${latestStatus.policy_hz??0} Hz`;$('timing').textContent=`推理与物理 ${latestStatus.inference_ms??0} ms`;$('age').textContent=`数据年龄 ${latestStatus.pose_age_ms??'—'} ms`;$('follow').disabled=!latestStatus.ready||!latestStatus.calibrated||latestStatus.fallen;$('calibrate').disabled=!latestStatus.ready;
// 真机开关只在 g1ra 后端有意义；开着的时用红色提示，避免误以为还在仿真。
const rb=$('robotToggle');
rb.style.display=latestStatus.backend==='g1ra'?'':'none';
rb.textContent=latestStatus.robot_enabled?'真机：开':'真机：关';
rb.className=latestStatus.robot_enabled?'stop':'';
rb.disabled=latestStatus.backend!=='g1ra'||!latestStatus.ready||(!latestStatus.robot_enabled&&!latestStatus.calibrated);
const real=latestStatus.backend==='g1ra';
$('pageTitle').textContent=real?'实机遥操 · HoloMotion → G1':'纯 MuJoCo 调试 · 摄像头 → HoloMotion';
document.title=real?'HoloMotion 实机遥操':'HoloMotion 纯 MuJoCo 调试';
$('modeHint').textContent=real?'此入口连接实机；本地仿真反馈仍保留实机对齐和断流保护。':'此入口只推进本地 MuJoCo，不连接机器人；开启摄像头后自动匹配骨架，可直接开始跟随。';
$('simEntry').setAttribute('aria-current',real?'false':'page');
$('realEntry').setAttribute('aria-current',real?'page':'false');
$('displaySource').value=latestStatus.display_source||'simulation';
$('displaySource').querySelector('option[value="robot"]').disabled=!latestStatus.robot_view_supported;
$('displaySource').disabled=!latestStatus.ready;
$('feedbackControl').hidden=!real;
$('feedbackSource').value=latestStatus.feedback_source||'robot';
$('feedbackSource').disabled=!latestStatus.ready||latestStatus.running||latestStatus.feedback_pending||latestStatus.align||['capturing','waiting_policy','moving','holding'].includes(latestStatus.alignment_status);
$('renderState').textContent=(latestStatus.render_source==='robot'?'实机位姿':'本地物理仿真')+(latestStatus.render_available?'':' · 暂无有效画面');
$('feedbackHint').textContent=real?`当前策略反馈：${latestStatus.feedback_source==='simulation'?'本地 MuJoCo（实机状态仍用于对齐和断流保护）':'实机回传'}。切换需先暂停并重新对齐。`:'本地 MuJoCo 物理仿真';
$('follow').disabled=$('follow').disabled||latestStatus.feedback_pending;
$('calibrate').textContent=real?'2 · 采集并对齐机器人（会运动）':'2 · 校准站姿';
$('calibrate').title=real?'冻结当前人体姿态，缓慢移动实机肩肘；完成后可开始跟随':'';
$('calibrate').disabled=!latestStatus.ready||(real&&(!latestStatus.state_valid||latestStatus.state_age_ms==null||latestStatus.state_age_ms>500||latestStatus.pose_age_ms==null||latestStatus.pose_age_ms>500));
$('pause').textContent=real?'暂停 / 停止对齐':'暂停仿真';
$('alignment').textContent=real&&latestStatus.alignment_error_deg!=null?`对齐误差 ${latestStatus.alignment_error_deg}° · 实测速度 ${latestStatus.alignment_speed_deg_s??'—'}°/s`:'';
if(latestStatus.error)notice(latestStatus.error);}catch(e){$('stage').textContent='无法连接本地服务，请在 VS Code 终端启动';}setTimeout(poll,500);}poll();
// 机器人画面是后端生成的 JPEG，连续请求实现刷新，不是浏览器本地仿真。
const robot=$('robot');function frame(){robot.src='/frame.jpg?t='+Date.now();}robot.onload=()=>setTimeout(frame,location.port==='8766'?16:80);robot.onerror=()=>setTimeout(frame,800);frame();
// 离开页面尝试通知暂停并关闭摄像头；后端还有 0.5 秒数据过期保护。
window.addEventListener('pagehide',()=>{navigator.sendBeacon('/api/control',new Blob([JSON.stringify({action:'pause'})],{type:'application/json'}));stream?.getTracks().forEach(t=>t.stop());});
