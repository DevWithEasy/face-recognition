const video = document.getElementById("video");

const startWebcam = () => {
    navigator.mediaDevices
        .getUserMedia({
            video: true,
            audio: false,
        })
        .then((stream) => {
            video.srcObject = stream;
        })
        .catch((error) => {
            console.error(error);
        });
};

const getLabelFaceDescriptions = () => {
    const labels = ["Busra", "Labib","Fairy","NurNahar","Robiul"];
    return Promise.all(
        labels.map(async (label) => {
            const descriptions = [];

            for (i = 1; i <= 1; i++) {
                const image = await faceapi.fetchImage(`./labels/${label}/${i}.jpg`);
                const detections = await faceapi
                    .detectSingleFace(image)
                    .withFaceLandmarks()
                    .withFaceDescriptor();
                descriptions.push(detections?.descriptor);
            }
            return new faceapi.LabeledFaceDescriptors(label, descriptions);
        })
    );
};

const faceRecognition = async () => {
    const LabeledFaceDescriptors = await getLabelFaceDescriptions();
    const faceMatcher = new faceapi.FaceMatcher(LabeledFaceDescriptors);

    video.addEventListener("play", () => {
        const canvas = faceapi.createCanvasFromMedia(video);

        document.body.append(canvas);

        const displaySize = { width: video.width, height: video.height };

        faceapi.matchDimensions(canvas, displaySize);

        setInterval(async () => {
            const detections = await faceapi
                .detectAllFaces(video)
                .withFaceLandmarks()
                .withFaceDescriptors();
                const resizedDetections = faceapi.resizeResults(detections,displaySize)

                canvas.getContext('2d').clearRect(0, 0,canvas.width, canvas.height)
        
                const results = resizedDetections.map((d)=>{
                    return faceMatcher.findBestMatch(d.descriptor)
                })
        
                results.forEach((result,i)=>{
                    const box = resizedDetections[i].detection.box
        
                    const drawBox = new faceapi.draw.DrawBox(box,{label: result})
                    drawBox.draw(canvas)
                })
        }, 100)
    });
};

Promise.all([
    faceapi.nets.ssdMobilenetv1.loadFromUri("./models"),
    faceapi.nets.faceLandmark68Net.loadFromUri("./models"),
    faceapi.nets.faceRecognitionNet.loadFromUri("./models"),
])
    .then(startWebcam)
    .then(faceRecognition)
    .catch(error =>console.error(error))
