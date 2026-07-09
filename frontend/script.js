const API_BASE = 'http://localhost:8000/api';

document.addEventListener('DOMContentLoaded', () => {
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const browseBtn = document.getElementById('browse-btn');
    const fileListContainer = document.getElementById('file-list-container');
    const fileList = document.getElementById('file-list');
    const clearBtn = document.getElementById('clear-btn');
    const uploadBtn = document.getElementById('upload-btn');
    const processingOverlay = document.getElementById('processing-overlay');
    const progressBar = document.getElementById('progress-bar');
    const processingText = document.getElementById('processing-text');
    const errorText = document.getElementById('error-text');
    const cancelBtn = document.getElementById('cancel-btn');
    const modalityBadge = document.getElementById('modality-badge');
    
    const uploadSection = document.getElementById('upload-section');
    const dashboardSection = document.getElementById('dashboard-section');
    const newScanBtn = document.getElementById('new-scan-btn');

    let selectedFile = null;

    // Prevent default drag behaviors
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, e => {
            e.preventDefault();
            e.stopPropagation();
        }, false);
    });

    ['dragenter', 'dragover'].forEach(eventName => {
        dropZone.addEventListener(eventName, () => dropZone.classList.add('dragover'), false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, () => dropZone.classList.remove('dragover'), false);
    });

    // Handle dropped files
    dropZone.addEventListener('drop', e => handleFile(e.dataTransfer.files[0]), false);
    browseBtn.addEventListener('click', () => fileInput.click());
    
    fileInput.addEventListener('change', function() {
        if (this.files.length > 0) handleFile(this.files[0]);
        this.value = null;
    });

    function handleFile(file) {
        if (!file) return;
        selectedFile = file;
        
        fileList.innerHTML = `
            <li class="file-item">
                <div class="file-info">
                    <span class="file-name">${file.name}</span>
                    <span class="file-size">${formatBytes(file.size)}</span>
                </div>
            </li>
        `;
        
        fileListContainer.style.display = 'block';
        dropZone.style.display = 'none';
    }

    function formatBytes(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024, i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + ['Bytes', 'KB', 'MB', 'GB'][i];
    }

    clearBtn.addEventListener('click', () => {
        selectedFile = null;
        fileListContainer.style.display = 'none';
        dropZone.style.display = 'block';
    });
    
    newScanBtn.addEventListener('click', () => {
        dashboardSection.style.display = 'none';
        uploadSection.style.display = 'block';
        clearBtn.click();
    });

    cancelBtn.addEventListener('click', () => {
        processingOverlay.style.display = 'none';
        document.querySelector('.spinner').style.display = 'block';
        errorText.style.display = 'none';
        cancelBtn.style.display = 'none';
        progressBar.style.width = '0%';
    });

    uploadBtn.addEventListener('click', async () => {
        if (!selectedFile) return;

        processingOverlay.style.display = 'flex';
        processingText.textContent = 'Uploading File...';
        progressBar.style.width = '10%';
        modalityBadge.textContent = 'Detecting...';
        
        const formData = new FormData();
        formData.append('file', selectedFile);

        try {
            // 1. Upload and detect modality
            const uploadRes = await fetch(`${API_BASE}/process`, {
                method: 'POST',
                body: formData
            });
            
            const uploadData = await uploadRes.json();
            if (!uploadRes.ok) throw new Error(uploadData.reason || uploadData.error || 'Upload failed');
            
            modalityBadge.textContent = `${uploadData.modality} Scan Detected`;
            document.getElementById('modality-badge-dashboard').textContent = `${uploadData.modality} Scan`;
            
            // 2. Poll for status
            pollStatus(uploadData.task_id);
            
        } catch (err) {
            showError(err.message);
        }
    });

    function pollStatus(taskId) {
        const interval = setInterval(async () => {
            try {
                const res = await fetch(`${API_BASE}/status/${taskId}`);
                const data = await res.json();
                
                if (data.state === 'segmenting') {
                    processingText.textContent = 'Segmenting Scan Data...';
                    progressBar.style.width = '40%';
                } else if (data.state === 'meshing') {
                    processingText.textContent = 'Generating 3D Models...';
                    progressBar.style.width = '70%';
                } else if (data.state === 'failed') {
                    clearInterval(interval);
                    showError(data.reason || 'Pipeline failed');
                } else if (data.state === 'complete') {
                    clearInterval(interval);
                    progressBar.style.width = '100%';
                    processingText.textContent = 'Complete!';
                    
                    setTimeout(() => loadDashboard(taskId), 500);
                }
            } catch (err) {
                clearInterval(interval);
                showError('Error checking status');
            }
        }, 1500);
    }
    
    function showError(msg) {
        document.querySelector('.spinner').style.display = 'none';
        processingText.textContent = 'Error';
        errorText.textContent = msg;
        errorText.style.display = 'block';
        cancelBtn.style.display = 'block';
        progressBar.style.width = '100%';
        progressBar.style.background = 'var(--danger)';
    }

    async function loadDashboard(taskId) {
        processingOverlay.style.display = 'none';
        uploadSection.style.display = 'none';
        dashboardSection.style.display = 'grid';
        
        // Fetch Results
        try {
            const res = await fetch(`${API_BASE}/results/${taskId}`);
            const scores = await res.json();
            
            document.getElementById('score-dice').textContent = scores.dice_score;
            document.getElementById('score-hausdorff').textContent = `${scores.hausdorff_distance} mm`;
            document.getElementById('score-jsw').textContent = `${scores.joint_space_width_mm} mm`;
            document.getElementById('score-maa').textContent = `${scores.mechanical_axis_angle} °`;
        } catch (e) {
            console.error("Failed to load scores", e);
        }
        
        // Init 3D Viewer
        if (window.initViewer) {
            window.initViewer(taskId);
        }
    }
});
