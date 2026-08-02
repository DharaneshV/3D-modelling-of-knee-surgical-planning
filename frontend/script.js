// Same-origin: the page is served by the backend, so the API is a relative
// path. A hardcoded host breaks the moment the page is opened from anything
// other than the machine running the server — a phone, most obviously.
const API_BASE = '/api';

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
    // Task currently shown on the dashboard. Tracked here rather than reusing
    // slice_viewer.js's currentTaskId so this file doesn't depend on load order.
    let dashboardTaskId = null;

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

    const arModal = document.getElementById('ar-modal');
    const arModelViewer = document.getElementById('ar-model-viewer');
    document.getElementById('ar-modal-close').addEventListener('click', () => {
        arModal.style.display = 'none';
        arModelViewer.src = '';
    });

    // --- Resection planning ---
    const resectionSliders = {
        'femur-depth-slider': ['femur-depth-label', v => `${v} mm`],
        'tibia-depth-slider': ['tibia-depth-label', v => `${v} mm`],
        'varus-slider': ['varus-label', v => `${v}°`],
        'slope-slider': ['slope-label', v => `${v}°`],
    };
    Object.entries(resectionSliders).forEach(([sliderId, [labelId, fmt]]) => {
        const el = document.getElementById(sliderId);
        if (el) el.addEventListener('input', e => {
            document.getElementById(labelId).textContent = fmt(e.target.value);
        });
    });

    function showResected(show) {
        window.activeViewports.forEach(vp => {
            vp.swapPart('femur_unknown.obj', show ? 'femur_resected.obj' : null);
            vp.swapPart('tibia_unknown.obj', show ? 'tibia_resected.obj' : null);
            vp.setBoneOpaque(show);
            vp.setExtraPart('tibial_tray.obj', show ? '#c8d0dc' : null);
            vp.setExtraPart('femoral_component.obj', show ? '#c8d0dc' : null);
        });

        // Articular cartilage sits on the surfaces being cut, so a resection
        // takes it with the bone — leaving it on screen would misrepresent the
        // result, and it also occludes the cut face. Driven through the part
        // checkboxes so the panel stays in sync with what is displayed.
        document.querySelectorAll('.part-controls label').forEach(row => {
            const name = row.querySelector('span:last-child').textContent;
            if (!name.includes('Cartilage')) return;
            const cb = row.querySelector('input');
            if (cb.checked === show) {
                cb.checked = !show;
                cb.dispatchEvent(new Event('change'));
            }
        });

        document.getElementById('restore-intact-btn').style.display = show ? 'block' : 'none';
    }

    const planBtn = document.getElementById('plan-resection-btn');
    if (planBtn) planBtn.addEventListener('click', async () => {
        if (!dashboardTaskId) return;
        planBtn.disabled = true;
        planBtn.textContent = 'Planning...';
        try {
            const q = new URLSearchParams({
                femur_depth_mm: document.getElementById('femur-depth-slider').value,
                tibia_depth_mm: document.getElementById('tibia-depth-slider').value,
                varus_deg: document.getElementById('varus-slider').value,
                slope_deg: document.getElementById('slope-slider').value,
            });
            const res = await fetch(`${API_BASE}/resect/${dashboardTaskId}?${q}`);
            const data = await res.json();
            if (!res.ok) throw new Error(data.detail || 'Resection failed');

            const body = document.getElementById('resection-body');
            body.innerHTML = '';
            Object.entries(data.resections).forEach(([bone, r]) => {
                const removed = r.removed_volume_mm3
                    ? `${(r.removed_volume_mm3 / 1000).toFixed(1)} cm³` : 'n/a';
                const tr = document.createElement('tr');
                tr.innerHTML = `<td>${bone} @ ${r.depth_mm}mm</td>` +
                    `<td>${r.cut_surface.ml_mm} mm</td>` +
                    `<td>${r.cut_surface.ap_mm} mm</td><td>${removed}</td>`;
                body.appendChild(tr);
            });
            const implants = data.implants || {};
            const labels = { femoral: 'Femoral component', tibial: 'Tibial tray' };
            Object.entries(implants).forEach(([kind, imp]) => {
                // Each component reports the number that governs its own risk:
                // the tray's is overhang, the femur's is AP headroom against
                // notching the anterior cortex.
                const note = kind === 'tibial'
                    ? `${imp.coverage_pct}% cover` +
                      (imp.max_overhang_mm != null ? `, ${imp.max_overhang_mm} mm overhang` : '')
                    : `${imp.ap_margin_mm} mm AP margin` +
                      (imp.ml_overhang ? ', ML overhangs' : '');
                const tr = document.createElement('tr');
                tr.innerHTML = `<td><strong>${labels[kind]} size ${imp.size}</strong>` +
                    `${imp.fit === 'fitted' ? '' : ' <em>(undersize)</em>'}</td>` +
                    `<td>${imp.ml_mm} mm</td><td>${imp.ap_mm} mm</td><td>${note}</td>`;
                body.appendChild(tr);
            });

            document.getElementById('resection-caveat').textContent = data.axis_note +
                ' Cut-surface dimensions are component sizing references only.' +
                (Object.keys(implants).length ? ' ' + data.implant_note : '');
            document.getElementById('resection-results').style.display = 'block';

            showResected(true);
        } catch (e) {
            console.error('Resection failed', e);
            document.getElementById('resection-caveat').textContent = `Resection failed: ${e.message}`;
            document.getElementById('resection-results').style.display = 'block';
        } finally {
            planBtn.disabled = false;
            planBtn.textContent = 'Plan Resection';
        }
    });

    const restoreBtn = document.getElementById('restore-intact-btn');
    if (restoreBtn) restoreBtn.addEventListener('click', () => showResected(false));

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
        processingText.textContent = `Processing: ${selectedFile.name}`;
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
            
            if (uploadData.cached) {
                processingText.textContent = 'Previously processed — loaded instantly.';
                progressBar.style.width = '100%';
            }
            
            // 2. Poll for status
            pollStatus(uploadData.task_id);
            
            // 3. Initialize slice viewer immediately
            if (window.initSliceViewer) {
                // Delay slightly to let the UI transition finish
                setTimeout(() => {
                    document.getElementById('dashboard-section').style.display = 'grid';
                    document.getElementById('upload-section').style.display = 'none';
                    initSliceViewer(uploadData.task_id);
                }, 500);
            }
            
        } catch (err) {
            showError(err.message);
        }
    });

    function pollStatus(taskId) {
        const interval = setInterval(async () => {
            try {
                const res = await fetch(`${API_BASE}/status/${taskId}`);
                const data = await res.json();
                
                if (data.laterality && data.laterality !== 'unknown') {
                    const latBadge = document.getElementById('laterality-badge');
                    latBadge.style.display = 'block';
                    latBadge.textContent = data.laterality === 'bilateral' ? 'Bilateral' : (data.laterality.charAt(0).toUpperCase() + data.laterality.slice(1) + ' Knee');
                }

                if (data.state === 'segmenting') {
                    processingText.textContent = 'Segmenting Scan Data...';
                    progressBar.style.width = '40%';
                    document.getElementById('report-placeholder-text').textContent = 'Segmenting scan data...';
                } else if (data.state === 'meshing') {
                    processingText.textContent = 'Generating 3D Models...';
                    progressBar.style.width = '70%';
                    document.getElementById('report-placeholder-text').textContent = 'Generating 3D models...';
                } else if (data.state === 'generating_report') {
                    processingText.textContent = 'Generating PDF Report...';
                    progressBar.style.width = '85%';
                    document.getElementById('report-placeholder-text').textContent = 'Generating report...';
                } else if (data.state === 'failed') {
                    clearInterval(interval);
                    showError(data.reason || 'Pipeline failed');
                    // Show failure in report panel as well
                    document.getElementById('report-placeholder-text').textContent = `Pipeline Failed: ${data.reason}`;
                    document.getElementById('report-placeholder-text').style.color = 'var(--danger)';
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
        dashboardTaskId = taskId;

        try {
            // Fetch Native Report Data
            const res = await fetch(`${API_BASE}/report/${taskId}/data`);
            const data = await res.json();
            
            // Populate Header
            document.getElementById('r-task-id').textContent = data.task_id;
            document.getElementById('r-scan-date').textContent = data.scan_date;
            document.getElementById('r-file-name').textContent = selectedFile ? selectedFile.name : 'Unknown';
            document.getElementById('r-modality').textContent = data.modality;
            document.getElementById('r-laterality').textContent = data.laterality;
            
            // Populate Metrics
            const tbody = document.getElementById('r-metrics-body');
            tbody.innerHTML = '';
            data.metrics.forEach(m => {
                const tr = document.createElement('tr');
                tr.innerHTML = `<td>${m.name}</td><td>${m.value}</td><td>${m.caveat}</td>`;
                tbody.appendChild(tr);
            });
            
            // Populate Impression
            document.getElementById('r-impression').textContent = data.impression;
            
            // Transition UI
            document.getElementById('report-placeholder').style.display = 'none';
            document.getElementById('report-container').style.display = 'block';
            
            const downloadBtn = document.getElementById('download-report-btn');
            downloadBtn.href = `${API_BASE}/report/${taskId}/pdf`;
            downloadBtn.style.display = 'block';

            // AR availability (MRI track only, when export_ar_glb produced a model)
            const arBtn = document.getElementById('view-ar-btn');
            try {
                const manifestRes = await fetch(`${API_BASE}/manifest/${taskId}`);
                const manifest = await manifestRes.json();
                if (manifest.ar_glb) {
                    arBtn.style.display = 'block';
                    arBtn.onclick = () => {
                        arModelViewer.src = `${API_BASE}/ar/${taskId}`;
                        arModal.style.display = 'flex';
                    };
                } else {
                    arBtn.style.display = 'none';
                }

                // The badge is hardcoded in the markup and otherwise only set
                // during upload, so a cached or revisited task showed the wrong
                // modality. The manifest is authoritative.
                if (manifest.modality) {
                    document.getElementById('modality-badge-dashboard').textContent =
                        `${manifest.modality} Scan`;
                }

                // Resection planning is MRI-only — it targets the CartiMorph
                // femur/tibia labels, which the CT track doesn't produce.
                const hasMriBones = manifest.parts.some(p => p.file === 'femur_unknown.obj');
                document.getElementById('resection-panel').style.display =
                    hasMriBones ? 'block' : 'none';
            } catch (e) {
                console.error("Failed to check AR availability", e);
                arBtn.style.display = 'none';
            }

        } catch (e) {
            console.error("Failed to load report data", e);
            document.getElementById('report-placeholder-text').textContent = 'Failed to load report data';
            document.getElementById('report-placeholder-text').style.color = 'var(--danger)';
        }
        
        // Init 3D Viewer
        if (window.initViewer) {
            window.initViewer(taskId);
        }
    }
});
