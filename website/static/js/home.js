// main.js

document.addEventListener('DOMContentLoaded', function() {
    const videoInput = document.getElementById('videoInput');
    const uploadArea = document.getElementById('uploadArea');
    const selectedFile = document.getElementById('selectedFile');
    const fileName = document.getElementById('fileName');
    const fileSize = document.getElementById('fileSize');
    const videoPreview = document.getElementById('videoPreview');
    const uploadForm = document.getElementById('uploadForm');
    const previewPlaceholder = document.querySelector('.preview-placeholder');
    const uploadProgress = document.getElementById("uploadProgress");
    const progressFill = document.getElementById("progressFill");
    const progressText = document.getElementById("progressText");



    uploadArea.addEventListener('dragover', (e) => {
        uploadArea.classList.add('drag-over');
    });

    uploadArea.addEventListener('dragleave', () => {
        uploadArea.classList.remove('drag-over');
    });

    uploadArea.addEventListener('drop', (e) => {
        uploadArea.classList.remove('drag-over');

        if (e.dataTransfer.files.length) {
            const file = e.dataTransfer.files[0];
            handleFileSelection(file);
        }
    });


    videoInput.addEventListener('change', function(e) { // 添加参数e
        console.log("CHANGE FIRED");
        console.log(e.target.files);
        const file = this.files[0]; // 使用this.files或者e.target.files
        if (file) {
            handleFileSelection(file);
        }
    });




    function handleFileSelection(file) {
        const allowedExtensions = ['mp4', 'avi', 'mov', 'mkv'];
        const fileExtension = file.name.split('.').pop().toLowerCase();

        if (!allowedExtensions.includes(fileExtension)) {
            showNotification('Invalid file type. Please upload MP4, AVI, MOV, or MKV video.', 'error');
            return;
        }

        if (file.size > 200 * 1024 * 1024) {
            showNotification('File size must be less than 200MB', 'error');
            return;
        }


        fileName.textContent = file.name;
        fileSize.textContent = formatFileSize(file.size);
        selectedFile.classList.add('show');


        const videoURL = URL.createObjectURL(file);
        videoPreview.src = videoURL;
        videoPreview.style.display = 'block';
        previewPlaceholder.style.display = 'none';

        videoPreview.onloadedmetadata = function() {

            videoPreview.currentTime = Math.min(videoPreview.duration * 0.1, 10);


            videoPreview.play().catch(e => {
                console.log('Autoplay prevented:', e.message);
                videoPreview.controls = true;
            });
        };

        videoPreview.onerror = function() {
            showNotification('Could not load video preview', 'error');
            videoPreview.style.display = 'none';
            previewPlaceholder.style.display = 'flex';
        };


        uploadProgress.style.display = "none";
        progressFill.style.width = "0%";
        progressText.textContent = "Uploading... 0%";
    }


    function formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }


    function showNotification(message, type = 'info') {

        const existingNotification = document.querySelector('.notification');
        if (existingNotification) {
            existingNotification.remove();
        }


        const notification = document.createElement('div');
        notification.className = `notification notification-${type}`;
        notification.innerHTML = `
            <i class="fas fa-${type === 'error' ? 'exclamation-circle' : 'check-circle'}"></i>
            <span>${message}</span>
        `;

        document.body.appendChild(notification);


        setTimeout(() => notification.classList.add('show'), 10);


        setTimeout(() => {
            notification.classList.remove('show');
            setTimeout(() => notification.remove(), 300);
        }, 4000);
    }

    uploadForm.addEventListener('submit', function(e) {

    if (!videoInput.files.length) {
        showNotification('Please select a video file first', 'error');
        e.preventDefault();
        return;
    }

    const analyzeBtn = document.querySelector('.analyze-btn');
    const originalText = analyzeBtn.innerHTML;

    analyzeBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Uploading...';
    analyzeBtn.disabled = true;


    uploadProgress.style.display = "block";

    let progress = 0;

    const interval = setInterval(() => {
        progress += 5;

        progressFill.style.width = progress + "%";
        progressText.textContent = "Uploading... " + progress + "%";

        if (progress >= 100) {
            clearInterval(interval);

            progressText.textContent = "Processing video...";
            analyzeBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Processing...';

            setTimeout(() => {
                analyzeBtn.innerHTML = originalText;
                analyzeBtn.disabled = false;
                showNotification('Video core_analysis started! Processing may take a few minutes.', 'success');
            }, 1500);
        }
    }, 100);
});


    const notificationStyle = document.createElement('style');
    notificationStyle.textContent = `
        .notification {
            position: fixed;
            top: 20px;
            right: 20px;
            background: rgba(16, 22, 30, 0.95);
            border-left: 4px solid #2ecc71;
            border-radius: 8px;
            padding: 16px 20px;
            display: flex;
            align-items: center;
            gap: 12px;
            color: white;
            box-shadow: 0 5px 15px rgba(0, 0, 0, 0.3);
            transform: translateX(100%);
            opacity: 0;
            transition: transform 0.3s, opacity 0.3s;
            z-index: 1000;
            min-width: 300px;
            max-width: 400px;
        }
        
        .notification.show {
            transform: translateX(0);
            opacity: 1;
        }
        
        .notification-error {
            border-left-color: #e74c3c;
        }
        
        .notification i {
            font-size: 20px;
        }
        
        .notification-error i {
            color: #e74c3c;
        }
        
        .notification-success i {
            color: #2ecc71;
        }
    `;
    document.head.appendChild(notificationStyle);



    function createTennisBall() {
        const ball = document.createElement('div');
        ball.className = 'tennis-ball';
        ball.innerHTML = '🎾';
        ball.style.position = 'fixed';
        ball.style.fontSize = '24px';
        ball.style.zIndex = '0';
        ball.style.left = Math.random() * 100 + 'vw';
        ball.style.top = '-50px';
        ball.style.opacity = '0.3';
        ball.style.pointerEvents = 'none';

        document.body.appendChild(ball);

        const duration = 3 + Math.random() * 4;
        const keyframes = [
            { transform: 'translateY(0) rotate(0deg)', opacity: 0.3 },
            { transform: `translateY(${window.innerHeight + 100}px) rotate(${360 * 3}deg)`, opacity: 0 }
        ];

        ball.animate(keyframes, {
            duration: duration * 1000,
            easing: 'cubic-bezier(0.4, 0, 0.2, 1)'
        });

        setTimeout(() => ball.remove(), duration * 1000);
    }

    setInterval(createTennisBall, 5000);
    setTimeout(createTennisBall, 1000);
});