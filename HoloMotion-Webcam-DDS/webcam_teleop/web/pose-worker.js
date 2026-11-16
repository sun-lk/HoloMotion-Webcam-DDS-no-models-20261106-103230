// 这是浏览器 Worker；使用本地 MediaPipe JS/WASM，避免页面交互被识别阻塞。
importScripts('./vendor/vision_bundle.js');
const {FilesetResolver,PoseLandmarker}=Vision;
let detector;
// 主线程发 init 初始化，随后发 frame（ImageBitmap + 毫秒时间戳）。
self.onmessage=async({data})=>{
  try{
    if(data.type==='init'){
      const files=await FilesetResolver.forVisionTasks(new URL('./vendor/wasm',self.location.href).href);
      // 当前识别使用浏览器 CPU；HoloMotion 的 GPU 推理在 WSL 后端，二者不同。
      detector=await PoseLandmarker.createFromOptions(files,{baseOptions:{modelAssetPath:new URL('./vendor/pose_landmarker_lite.task',self.location.href).href,delegate:'CPU'},runningMode:'VIDEO',numPoses:1,minPoseDetectionConfidence:.6,minPosePresenceConfidence:.6,minTrackingConfidence:.6});
      self.postMessage({type:'ready'});
    }else if(data.type==='frame'){
      try{
        // VIDEO 模式时间戳来自 performance.now()，用于同一浏览器内的帧顺序。
        const result=detector.detectForVideo(data.bitmap,data.timestamp);
        // landmarks 用于二维叠图，worldLandmarks 用于三维动作映射；只取第一个人。
        self.postMessage({type:'pose',timestamp:data.timestamp,landmarks:result.landmarks[0]||null,world:result.worldLandmarks[0]||null});
      // 及时释放传入的图像对象，避免摄像头长时间运行造成内存积累。
      }finally{data.bitmap.close();}
    }
  }catch(error){self.postMessage({type:'error',message:String(error)});}
};
