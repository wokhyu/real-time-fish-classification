document.addEventListener('DOMContentLoaded', function() {
    console.log('🐟 FishVision AI Started');

    // UI Elements
    const imageBtn = document.getElementById('imageBtn');
    const webcamBtn = document.getElementById('webcamBtn');
    const uploadSections = document.querySelectorAll('.upload-section');
    const imageUpload = document.getElementById('imageUpload');
    const webcamSection = document.getElementById('webcamSection');
    const resultsSection = document.getElementById('resultsSection');
    const imageResults = document.getElementById('imageResults');
    const inputPreview = document.getElementById('inputPreview');
    const outputPreview = document.getElementById('outputPreview');
    const speciesInfo = document.getElementById('speciesInfo');
    const downloadBtn = document.getElementById('downloadBtn');

    // Input elements
    const imageInput = document.getElementById('imageInput');
    const webcam = document.getElementById('webcam');
    const webcamOverlay = document.getElementById('webcamOverlay');
    const webcamPlaceholder = document.getElementById('webcamPlaceholder');
    const startWebcamBtn = document.getElementById('startWebcamBtn');
    const analyzeImageBtn = document.getElementById('analyzeImageBtn');

    // Variables
    let currentStream = null;
    let selectedFile = null;
    let webcamRealtimeActive = false;
    let webcamPaused = false;
    let webcamProcessingInterval = null;
    let isProcessingFrame = false;

    // API URLs
    const IMAGE_API_URL = "YOUR_API_URL/analyze/image";
    const WEBCAM_API_URL = "YOUR_API_URL/webcam/stream";
    // Event listeners
    imageBtn.addEventListener('click', () => {
        console.log('📸 Image button clicked');
        showSection(imageUpload);
    });
    
    webcamBtn.addEventListener('click', () => {
        console.log('📹 Webcam button clicked');
        showSection(webcamSection);
    });
    
    imageInput.addEventListener('change', handleImageUpload);
    setupDragDrop(imageUpload, 'image/*', handleImageUpload);
    startWebcamBtn.addEventListener('click', toggleWebcamRealtime);
    analyzeImageBtn.addEventListener('click', analyzeImage);
    downloadBtn.addEventListener('click', downloadResults);

    function showSection(sectionToShow) {
        uploadSections.forEach(section => section.classList.add('hidden'));
        resultsSection.classList.add('hidden');
        if (currentStream) stopWebcamRealtime();
        sectionToShow.classList.remove('hidden');
    }

    function setupDragDrop(container, acceptType, handler) {
        const dropArea = container.querySelector('.border-2');
        
        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
            dropArea.addEventListener(eventName, (e) => {
                e.preventDefault();
                e.stopPropagation();
            }, false);
        });

        ['dragenter', 'dragover'].forEach(eventName => {
            dropArea.addEventListener(eventName, () => {
                dropArea.classList.add('border-blue-500');
            }, false);
        });

        ['dragleave', 'drop'].forEach(eventName => {
            dropArea.addEventListener(eventName, () => {
                dropArea.classList.remove('border-blue-500');
            }, false);
        });

        dropArea.addEventListener('drop', function(e) {
            const files = e.dataTransfer.files;
            if (files.length > 0 && files[0].type.match(acceptType)) {
                const fakeEvent = { target: { files: [files[0]] } };
                handler(fakeEvent);
            }
        }, false);
    }

    function handleImageUpload(e) {
    if (e.target.files && e.target.files[0]) {
        selectedFile = e.target.files[0];
        console.log('✅ Image selected:', selectedFile.name);
        
        const reader = new FileReader();
        const fileSize = (selectedFile.size / 1024 / 1024).toFixed(2);
        
        reader.onload = function(event) {
            // ✅ Xóa toàn bộ nội dung cũ
            inputPreview.innerHTML = '';
            
            // Tạo container cho ảnh
            const imgContainer = document.createElement('div');
            imgContainer.className = 'w-full h-full flex items-center justify-center';
            
            // Tạo ảnh
            const img = document.createElement('img');
            img.src = event.target.result;
            img.className = 'max-w-full max-h-full object-contain rounded-lg';
            
            imgContainer.appendChild(img);
            inputPreview.appendChild(imgContainer);
            
            // ✅ Tạo file info BÊN NGOÀI inputPreview
            const previewSection = inputPreview.parentElement;
            
            // Xóa file info cũ nếu có
            const oldFileInfo = previewSection.querySelector('.file-info-external');
            if (oldFileInfo) oldFileInfo.remove();
            
            // Tạo file info mới
            const fileInfo = document.createElement('div');
            fileInfo.className = 'file-info-external mt-3 text-center';
            fileInfo.innerHTML = `
                <p class="text-sm text-gray-600 font-medium">
                    📁 ${selectedFile.name}
                </p>
                <p class="text-xs text-gray-500 mt-1">
                    ${fileSize} MB
                </p>
            `;
            
            // Thêm vào sau inputPreview
            previewSection.appendChild(fileInfo);
            
            analyzeImageBtn.disabled = false;
        };
        reader.readAsDataURL(selectedFile);
    }
}

    // ==================== IMAGE ANALYSIS ====================
    async function analyzeImage() {
        if (!selectedFile) return;

        console.log('🔍 Starting image analysis...');
        analyzeImageBtn.disabled = true;
        analyzeImageBtn.innerHTML = '<i data-feather="loader" class="animate-spin"></i> Analyzing...';
        feather.replace();

        const progressDiv = createProgressBar('📤 Đang tải ảnh lên...');

        try {
            const formData = new FormData();
            formData.append('file', selectedFile);

            updateProgress(progressDiv, 50, 'Đang tải lên...');
            
            const response = await fetch(IMAGE_API_URL, {
                method: "POST",
                body: formData
            });

            updateProgress(progressDiv, 100, '⚙️ Đang phân tích...');

            if (!response.ok) {
                const errorText = await response.text();
                throw new Error(`Server error ${response.status}: ${errorText}`);
            }

            const result = await response.json();
            console.log("✅ Image API Response:", result);

            removeProgressBar(progressDiv);
            displayImageResults(result);
            
        } catch (error) {
            console.error("❌ Analysis error:", error);
            removeProgressBar(progressDiv);
            
            outputPreview.innerHTML = `
                <div class="text-center text-red-600 p-4">
                    <p class="font-bold mb-2">❌ Lỗi kết nối Backend</p>
                    <p class="text-sm">Đảm bảo Backend đang chạy tại:</p>
                    <p class="text-xs font-mono bg-red-50 p-2 rounded mt-2">http://127.0.0.1:8000</p>
                    <p class="text-xs mt-2">Chi tiết: ${error.message}</p>
                </div>
            `;
            resultsSection.classList.remove('hidden');
        } finally {
            analyzeImageBtn.disabled = false;
            analyzeImageBtn.innerHTML = 'Analyze Image';
            feather.replace();
        }
    }

    function displayImageResults(results) {
        resultsSection.classList.remove('hidden');
        imageResults.classList.remove('hidden');

        outputPreview.innerHTML = '';
        
        // Hiển thị ảnh annotated
        if (results.annotated_image) {
            const img = document.createElement('img');
            img.src = results.annotated_image;
            img.className = 'max-w-full max-h-full object-contain rounded-lg shadow-lg';
            outputPreview.appendChild(img);
        } else {
            const inputImg = inputPreview.querySelector('img');
            if (inputImg) {
                const img = document.createElement('img');
                img.src = inputImg.src;
                img.className = 'max-w-full max-h-full object-contain';
                outputPreview.appendChild(img);
            }
        }

        // Hiển thị kết quả phát hiện nhiều cá
        if (results.total_fish > 0) {
            let fishCardsHTML = '';
            
            results.detections.forEach((det, index) => {
                const wikiLink = det.wikiLink ? `
                    <a href="${det.wikiLink}" target="_blank" rel="noopener noreferrer" 
                       class="inline-flex items-center gap-2 bg-gradient-to-r from-blue-500 to-blue-600 hover:from-blue-600 hover:to-blue-700 text-white px-4 py-2 rounded-lg transition-all shadow-md hover:shadow-lg text-sm w-full justify-center mt-2">
                        <svg class="w-4 h-4" fill="currentColor" viewBox="0 0 24 24">
                            <path d="M12.87 15.07l-2.54-2.51.03-.03c1.74-1.94 2.98-4.17 3.71-6.53H17V4h-7V2H8v2H1v1.99h11.17C11.5 7.92 10.44 9.75 9 11.35 8.07 10.32 7.3 9.19 6.69 8h-2c.73 1.63 1.73 3.17 2.98 4.56l-5.09 5.02L4 19l5-5 3.11 3.11.76-2.04zM18.5 10h-2L12 22h2l1.12-3h4.75L21 22h2l-4.5-12zm-2.62 7l1.62-4.33L19.12 17h-3.24z"/>
                        </svg>
                        <span class="font-semibold">Wikipedia</span>
                        <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14"/>
                        </svg>
                    </a>
                ` : '';
                
                fishCardsHTML += `
                    <div class="bg-white p-4 rounded-lg shadow species-card border-l-4 border-blue-500">
                        <div class="flex items-center justify-between mb-2">
                            <h4 class="font-bold text-lg text-blue-700">🐟 Fish #${det.fish_id}</h4>
                            <span class="bg-blue-600 text-white px-3 py-1 rounded-full text-sm font-bold">
                                ${Math.round(det.confidence * 100)}%
                            </span>
                        </div>
                        <p class="text-gray-800 font-semibold mb-1">${det.species}</p>
                        <p class="text-gray-500 text-sm italic mb-3">${det.scientificName}</p>
                        <div class="flex items-center mb-3">
                            <span class="text-gray-700 text-sm mr-2">Confidence:</span>
                            <div class="flex-1 bg-gray-200 rounded-full h-2 overflow-hidden">
                                <div class="bg-blue-600 h-2 rounded-full transition-all" style="width: ${det.confidence * 100}%"></div>
                            </div>
                        </div>
                        ${wikiLink}
                    </div>
                `;
            });
            
            speciesInfo.innerHTML = `
                <div class="bg-gradient-to-r from-blue-500 to-blue-600 text-white p-4 rounded-lg shadow-lg mb-4">
                    <div class="flex items-center justify-between">
                        <div>
                            <p class="text-sm opacity-90">Total Fish Detected</p>
                            <p class="text-3xl font-bold">${results.total_fish}</p>
                        </div>
                        <svg class="w-16 h-16 opacity-80" fill="currentColor" viewBox="0 0 20 20">
                            <path d="M2 5a2 2 0 012-2h7a2 2 0 012 2v4a2 2 0 01-2 2H9l-3 3v-3H4a2 2 0 01-2-2V5z"/>
                            <path d="M15 7v2a4 4 0 01-4 4H9.828l-1.766 1.767c.28.149.599.233.938.233h2l3 3v-3h2a2 2 0 002-2V9a2 2 0 00-2-2h-1z"/>
                        </svg>
                    </div>
                </div>
                <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
                    ${fishCardsHTML}
                </div>
            `;
        } else {
            speciesInfo.innerHTML = `
                <div class="bg-yellow-50 border-l-4 border-yellow-400 p-4 rounded-lg">
                    <p class="text-yellow-800 font-semibold">⚠️ ${results.message}</p>
                </div>
            `;
        }

        downloadBtn.classList.remove('hidden');
    }

    // ==================== WEBCAM REAL-TIME ====================
    async function toggleWebcamRealtime() {
        if (webcamRealtimeActive) {
            stopWebcamRealtime();
        } else {
            await startWebcamRealtime();
        }
    }

    async function startWebcamRealtime() {
        try {
            console.log('📹 Starting webcam...');
            const stream = await navigator.mediaDevices.getUserMedia({ 
                video: { width: 640, height: 480 }, 
                audio: false 
            });
            
            currentStream = stream;
            webcam.srcObject = stream;
            webcamPlaceholder.classList.add('hidden');
            webcam.classList.remove('hidden');
            webcamOverlay.classList.remove('hidden');

            webcam.addEventListener('loadedmetadata', () => {
                webcamOverlay.width = webcam.videoWidth;
                webcamOverlay.height = webcam.videoHeight;
            });

            startWebcamBtn.innerHTML = '<i data-feather="square"></i><span>Stop</span>';
            startWebcamBtn.classList.remove('bg-blue-600', 'hover:bg-blue-700');
            startWebcamBtn.classList.add('bg-red-600', 'hover:bg-red-700');
            feather.replace();

            webcamRealtimeActive = true;
            webcamPaused = false;

            createWebcamControlPanel();
            createWebcamInfoPanel();

            webcamProcessingInterval = setInterval(processWebcamFrame, 300);

        } catch (err) {
            console.error("❌ Webcam error:", err);
            alert("Không thể truy cập webcam. Vui lòng cấp quyền camera.");
        }
    }

    function createWebcamControlPanel() {
        const webcamContainer = webcamSection.querySelector('.bg-white.rounded-lg.shadow-lg');
        
        let controlPanel = webcamContainer.querySelector('.webcam-control-panel');
        if (controlPanel) controlPanel.remove();

        controlPanel = document.createElement('div');
        controlPanel.className = 'webcam-control-panel mt-4 flex gap-3 justify-center';
        controlPanel.innerHTML = `
            <button id="pauseWebcamBtn" class="flex-1 bg-yellow-500 hover:bg-yellow-600 text-white px-4 py-2 rounded-lg font-medium transition-colors flex items-center justify-center gap-2">
                <i data-feather="pause"></i>
                <span>Pause</span>
            </button>
            <button id="resumeWebcamBtn" class="flex-1 bg-green-500 hover:bg-green-600 text-white px-4 py-2 rounded-lg font-medium transition-colors flex items-center justify-center gap-2 hidden">
                <i data-feather="play"></i>
                <span>Resume</span>
            </button>
        `;
        
        webcamContainer.appendChild(controlPanel);
        feather.replace();

        document.getElementById('pauseWebcamBtn').addEventListener('click', pauseWebcam);
        document.getElementById('resumeWebcamBtn').addEventListener('click', resumeWebcam);
    }

    function createWebcamInfoPanel() {
        const webcamContainer = webcamSection.querySelector('.bg-white.rounded-lg.shadow-lg');
        
        let infoPanel = webcamContainer.querySelector('.webcam-info-panel');
        if (infoPanel) infoPanel.remove();

        infoPanel = document.createElement('div');
        infoPanel.className = 'webcam-info-panel mt-4';
        infoPanel.innerHTML = `
            <div class="bg-blue-50 border-l-4 border-blue-400 p-4 rounded-lg">
                <div class="flex items-center mb-2">
                    <span class="w-3 h-3 bg-green-500 rounded-full mr-2 animate-ping"></span>
                    <p class="text-sm font-semibold text-blue-800">🔴 Real-time Detection Active</p>
                </div>
                <p class="text-xs text-blue-700">
                    Hệ thống đang phân tích liên tục. Kết quả sẽ hiển thị khi phát hiện cá.
                </p>
            </div>
            <div id="webcamDetectionInfo" class="mt-4"></div>
        `;
        
        webcamContainer.appendChild(infoPanel);
    }

    function pauseWebcam() {
        webcamPaused = true;
        document.getElementById('pauseWebcamBtn').classList.add('hidden');
        document.getElementById('resumeWebcamBtn').classList.remove('hidden');

        const statusDiv = document.querySelector('.webcam-info-panel .bg-blue-50');
        if (statusDiv) {
            statusDiv.className = 'bg-yellow-50 border-l-4 border-yellow-400 p-4 rounded-lg';
            statusDiv.innerHTML = `
                <div class="flex items-center mb-2">
                    <span class="w-3 h-3 bg-yellow-500 rounded-full mr-2"></span>
                    <p class="text-sm font-semibold text-yellow-800">⏸️ Paused</p>
                </div>
                <p class="text-xs text-yellow-700">Click "Resume" để tiếp tục.</p>
            `;
        }
    }

    function resumeWebcam() {
        webcamPaused = false;
        document.getElementById('pauseWebcamBtn').classList.remove('hidden');
        document.getElementById('resumeWebcamBtn').classList.add('hidden');

        const statusDiv = document.querySelector('.webcam-info-panel .bg-yellow-50');
        if (statusDiv) {
            statusDiv.className = 'bg-blue-50 border-l-4 border-blue-400 p-4 rounded-lg';
            statusDiv.innerHTML = `
                <div class="flex items-center mb-2">
                    <span class="w-3 h-3 bg-green-500 rounded-full mr-2 animate-ping"></span>
                    <p class="text-sm font-semibold text-blue-800">🔴 Real-time Detection Active</p>
                </div>
                <p class="text-xs text-blue-700">Hệ thống đang phân tích liên tục.</p>
            `;
        }
    }

    function stopWebcamRealtime() {
        console.log('⏹️ Stopping webcam...');
        
        if (webcamProcessingInterval) {
            clearInterval(webcamProcessingInterval);
            webcamProcessingInterval = null;
        }

        if (currentStream) {
            currentStream.getTracks().forEach(track => track.stop());
            currentStream = null;
            webcam.srcObject = null;
        }

        webcam.classList.add('hidden');
        webcamOverlay.classList.add('hidden');
        webcamPlaceholder.classList.remove('hidden');

        startWebcamBtn.innerHTML = '<i data-feather="play"></i><span>Start Real-time</span>';
        startWebcamBtn.classList.remove('bg-red-600', 'hover:bg-red-700');
        startWebcamBtn.classList.add('bg-blue-600', 'hover:bg-blue-700');
        feather.replace();

        const controlPanel = document.querySelector('.webcam-control-panel');
        const infoPanel = document.querySelector('.webcam-info-panel');
        if (controlPanel) controlPanel.remove();
        if (infoPanel) infoPanel.remove();

        webcamRealtimeActive = false;
        webcamPaused = false;
    }

    async function processWebcamFrame() {
        if (!webcamRealtimeActive || !currentStream || webcamPaused) return;

        const canvas = document.createElement('canvas');
        canvas.width = webcam.videoWidth;
        canvas.height = webcam.videoHeight;
        const ctx = canvas.getContext('2d');
        ctx.drawImage(webcam, 0, 0, canvas.width, canvas.height);

        canvas.toBlob(async (blob) => {
            try {
                const formData = new FormData();
                formData.append('file', blob, 'webcam-frame.jpg');

                const response = await fetch(WEBCAM_API_URL, {
                    method: "POST",
                    body: formData
                });

                if (!response.ok) return;

                const result = await response.json();

                if (result.frame) {
                    const img = new Image();
                    img.onload = function() {
                        const ctx = webcamOverlay.getContext('2d');
                        ctx.clearRect(0, 0, webcamOverlay.width, webcamOverlay.height);
                        ctx.drawImage(img, 0, 0, webcamOverlay.width, webcamOverlay.height);
                    };
                    img.src = result.frame;
                }

                if (result.detection) {
                    updateWebcamDetectionUI(result);
                }

            } catch (error) {
                // Silent fail
            }
        }, 'image/jpeg', 0.65);
    }

    function updateWebcamDetectionUI(result) {
        const det = result.detection;
        const detectionInfoDiv = document.getElementById('webcamDetectionInfo');
        
        if (!detectionInfoDiv) return;

        const wikiLink = det.species && det.species !== "No Fish Detected" && det.species !== "N/A"
            ? `https://en.wikipedia.org/wiki/${det.species.replace(/ /g, "_")}`
            : null;

        detectionInfoDiv.innerHTML = `
            <div class="bg-green-50 border-2 border-green-400 p-4 rounded-lg shadow-md">
                <div class="flex items-center justify-between mb-3">
                    <div class="flex items-center">
                        <span class="w-3 h-3 bg-green-500 rounded-full mr-2 animate-ping"></span>
                        <h4 class="font-bold text-lg text-green-700">🐟 ${det.species}</h4>
                    </div>
                    <span class="bg-green-600 text-white px-3 py-1 rounded-full text-sm font-bold">
                        ${Math.round(det.confidence * 100)}%
                    </span>
                </div>
                
                <div class="mb-3">
                    <div class="w-full bg-gray-200 rounded-full h-3 overflow-hidden">
                        <div class="bg-gradient-to-r from-green-500 to-green-600 h-3 rounded-full transition-all" 
                             style="width: ${det.confidence * 100}%"></div>
                    </div>
                </div>

                ${wikiLink ? `
                    <a href="${wikiLink}" target="_blank" rel="noopener noreferrer" 
                       class="inline-flex items-center gap-2 bg-gradient-to-r from-blue-500 to-blue-600 hover:from-blue-600 hover:to-blue-700 text-white px-4 py-2 rounded-lg transition-all shadow-md hover:shadow-lg text-sm w-full justify-center mt-2">
                        <span class="font-semibold">View on Wikipedia</span>
                        <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14"/>
                        </svg>
                    </a>
                ` : ''}
            </div>
        `;
    }

    // ==================== UTILITY FUNCTIONS ====================
    function createProgressBar(message) {
        const progressDiv = document.createElement('div');
        progressDiv.className = 'fixed top-4 right-4 bg-white p-4 rounded-lg shadow-2xl z-50 border-2 border-blue-500';
        progressDiv.style.minWidth = '320px';
        progressDiv.innerHTML = `
            <div class="mb-2">
                <p class="text-sm font-semibold text-gray-800">${message}</p>
            </div>
            <div class="w-full bg-gray-200 rounded-full h-4 mb-2 overflow-hidden">
                <div class="progress-bar-fill bg-gradient-to-r from-blue-500 to-blue-600 h-4 rounded-full transition-all duration-300" style="width: 0%"></div>
            </div>
            <div class="flex justify-between text-xs text-gray-600">
                <span class="progress-percent">0%</span>
                <span class="progress-status">Đang chuẩn bị...</span>
            </div>
        `;
        document.body.appendChild(progressDiv);
        return progressDiv;
    }

    function updateProgress(progressDiv, percent, status) {
        if (!progressDiv) return;
        const fill = progressDiv.querySelector('.progress-bar-fill');
        const percentText = progressDiv.querySelector('.progress-percent');
        const statusText = progressDiv.querySelector('.progress-status');
        
        if (fill) fill.style.width = `${percent}%`;
        if (percentText) percentText.textContent = `${percent}%`;
        if (statusText) statusText.textContent = status;
    }

    function removeProgressBar(progressDiv) {
        if (progressDiv && progressDiv.parentNode) {
            progressDiv.style.opacity = '0';
            progressDiv.style.transition = 'opacity 0.3s';
            setTimeout(() => {
                if (progressDiv.parentNode) progressDiv.remove();
            }, 300);
        }
    }

    function downloadResults() {
        const notification = document.createElement('div');
        notification.className = 'fixed top-4 right-4 bg-green-500 text-white px-6 py-3 rounded-lg shadow-lg z-50';
        notification.textContent = '✅ Chức năng download sẽ được triển khai!';
        document.body.appendChild(notification);
        setTimeout(() => notification.remove(), 3000);
    }
    
    window.addEventListener('beforeunload', () => {
        if (webcamRealtimeActive) stopWebcamRealtime();
    });
    
    feather.replace();
});
