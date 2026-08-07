// API_BASE is declared once in slice_viewer.js, which loads before this file
// and shares the same top-level scope — see the comment there.

// Escapes text for safe innerHTML interpolation. Most call sites below
// interpolate our own server-computed numbers/labels, not attacker input, but
// file.name in handleFile() genuinely is user-controlled (a crafted filename
// like "<img src=x onerror=...>.nii.gz" would otherwise inject into the DOM),
// and the rest are consolidated onto the same helper rather than trusting
// each call site to individually reason about whether its value is safe.
function escapeHtml(str) {
    const div = document.createElement('div');
    div.textContent = String(str);
    return div.innerHTML;
}

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
    // pollStatus's setInterval handle. "Upload New Scan" used to leave this
    // running: if the abandoned pipeline finished afterward, its callback
    // still fired and silently overwrote dashboardTaskId/the 3D viewer with
    // the OLD task's data out from under whatever the user had since started.
    let activePollInterval = null;

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
                    <span class="file-name">${escapeHtml(file.name)}</span>
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
        uploadBtn.disabled = false;
    });
    
    newScanBtn.addEventListener('click', () => {
        if (activePollInterval) {
            clearInterval(activePollInterval);
            activePollInterval = null;
        }
        // Full teardown, not just stopping the render loop — the previous
        // task's WebGL resources should not keep existing once its dashboard
        // is gone, and this "Upload New Scan" click may be the only signal
        // that the user is done with them.
        if (window.activeViewports) {
            window.activeViewports.forEach(vp => vp.dispose());
            window.activeViewports = [];
        }
        dashboardTaskId = null;

        dashboardSection.style.display = 'none';
        uploadSection.style.display = 'block';
        clearBtn.click();
    });

    const arModal = document.getElementById('ar-modal');
    const arModelViewer = document.getElementById('ar-model-viewer');
    const arModalClose = document.getElementById('ar-modal-close');
    let arModalTrigger = null;  // element to return focus to on close

    function closeArModal() {
        arModal.style.display = 'none';
        arModelViewer.src = '';
        if (arModalTrigger) {
            arModalTrigger.focus();
            arModalTrigger = null;
        }
    }

    function openArModal(src) {
        arModalTrigger = document.activeElement;
        arModelViewer.src = src;
        arModal.style.display = 'flex';
        arModalClose.focus();
    }

    arModalClose.addEventListener('click', closeArModal);

    // Escape-to-close and returning focus on exit are both missing: the modal
    // had no role="dialog"/aria-modal (set in index.html), no keyboard escape
    // route, and a keyboard user had to tab blindly to find Close.
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape' && arModal.style.display !== 'none') {
            closeArModal();
        }
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

    // Remembers each cartilage checkbox's state from just before a resection
    // forced it off, so restoring intact anatomy restores what the user
    // actually had chosen rather than unconditionally re-checking everything
    // — a user who had deliberately hidden cartilage before ever planning a
    // resection was otherwise having that choice silently overwritten.
    let cartilagePreResectionState = null;

    function showResected(show, availableImplants = {}) {
        window.activeViewports.forEach(vp => {
            vp.swapPart('femur_unknown.obj', show ? 'femur_resected.obj' : null);
            vp.swapPart('tibia_unknown.obj', show ? 'tibia_resected.obj' : null);
            vp.setBoneOpaque(show);
            // Only display an implant if THIS resection actually produced one:
            // the file on disk can be stale (a fit that succeeded at a
            // previous slider setting but failed at the current one is never
            // deleted), so presence in the current response's `implants` —
            // not presence of the file — decides whether to show it.
            vp.setExtraPart('tibial_tray.obj', show && availableImplants.tibial ? '#c8d0dc' : null);
            vp.setExtraPart('femoral_component.obj', show && availableImplants.femoral ? '#c8d0dc' : null);
        });

        // Articular cartilage sits on the surfaces being cut, so a resection
        // takes it with the bone — leaving it on screen would misrepresent the
        // result, and it also occludes the cut face. Driven through the part
        // checkboxes so the panel stays in sync with what is displayed, and
        // disabled while resected so re-checking one can't put cartilage back
        // through the open cut face the resection just made.
        const cartilageCheckboxes = [...document.querySelectorAll('.part-controls label')]
            .filter(row => row.querySelector('span:last-child').textContent.includes('Cartilage'))
            .map(row => row.querySelector('input'));

        if (show) {
            cartilagePreResectionState = new Map(cartilageCheckboxes.map(cb => [cb, cb.checked]));
            cartilageCheckboxes.forEach(cb => {
                if (cb.checked) {
                    cb.checked = false;
                    cb.dispatchEvent(new Event('change'));
                }
                cb.disabled = true;
            });
        } else {
            cartilageCheckboxes.forEach(cb => {
                cb.disabled = false;
                const wasChecked = cartilagePreResectionState?.get(cb) ?? true;
                if (cb.checked !== wasChecked) {
                    cb.checked = wasChecked;
                    cb.dispatchEvent(new Event('change'));
                }
            });
            cartilagePreResectionState = null;
        }

        document.getElementById('restore-intact-btn').style.display = show ? 'block' : 'none';
    }

    const planBtn = document.getElementById('plan-resection-btn');
    if (planBtn) planBtn.addEventListener('click', async () => {
        if (!dashboardTaskId || planBtn.disabled) return;

        const body = document.getElementById('resection-body');
        // Cleared unconditionally, before the request, not just on success —
        // a failed re-plan after a prior successful one used to leave the old
        // cut-surface/implant rows on screen at the same time as the "failed"
        // caveat, a misleading mixed state.
        body.innerHTML = '';
        document.getElementById('resection-results').style.display = 'none';

        planBtn.disabled = true;
        // The request takes ~24s (box-cut resection + two implant fits + AR
        // export) with nothing but this label to show for it otherwise —
        // confirmed by direct timing. btn-loading adds a CSS pulse so the
        // wait doesn't read as a frozen page.
        planBtn.classList.add('btn-loading');
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

            // `resections` holds two different shapes. A single-plane cut
            // (always the tibia; the femur only when its box preparation fell
            // back) carries depth_mm/cut_surface/volumes. The femoral five-cut
            // box carries preparation/distal_depth_mm/component_size and has NO
            // cut_surface — there are five cut faces, not one, so a single
            // ML/AP pair would be meaningless. Reading r.cut_surface.ml_mm
            // unconditionally threw a TypeError on the femur row, which aborted
            // the whole handler: the table showed only the tibia and
            // showResected() never ran, so the 3D view silently stayed intact
            // after a "successful" resection.
            Object.entries(data.resections).forEach(([bone, r]) => {
                const removed = r.removed_volume_mm3
                    ? `${(r.removed_volume_mm3 / 1000).toFixed(1)} cm³` : 'n/a';
                const tr = document.createElement('tr');
                if (r.cut_surface) {
                    tr.innerHTML = `<td>${escapeHtml(bone)} @ ${r.depth_mm}mm</td>` +
                        `<td>${r.cut_surface.ml_mm} mm</td>` +
                        `<td>${r.cut_surface.ap_mm} mm</td><td>${removed}</td>`;
                } else {
                    tr.innerHTML = `<td>${escapeHtml(bone)} @ ${r.distal_depth_mm}mm</td>` +
                        `<td colspan="2">${escapeHtml(r.preparation || 'prepared')}</td>` +
                        `<td>${removed}</td>`;
                }
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
                tr.innerHTML = `<td><strong>${escapeHtml(labels[kind])} size ${imp.size}</strong>` +
                    `${imp.fit === 'fitted' ? '' : ' <em>(undersize)</em>'}</td>` +
                    `<td>${imp.ml_mm} mm</td><td>${imp.ap_mm} mm</td><td>${escapeHtml(note)}</td>`;
                body.appendChild(tr);
            });

            document.getElementById('resection-caveat').textContent = data.axis_note +
                ' Cut-surface dimensions are component sizing references only.' +
                (Object.keys(implants).length ? ' ' + data.implant_note : '');
            document.getElementById('resection-results').style.display = 'block';

            showResected(true, implants);
        } catch (e) {
            console.error('Resection failed', e);
            document.getElementById('resection-caveat').textContent = `Resection failed: ${e.message}`;
            document.getElementById('resection-results').style.display = 'block';
        } finally {
            planBtn.disabled = false;
            planBtn.classList.remove('btn-loading');
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
        uploadBtn.disabled = false;
    });

    uploadBtn.addEventListener('click', async () => {
        if (!selectedFile || uploadBtn.disabled) return;
        uploadBtn.disabled = true;

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
        activePollInterval = setInterval(async () => {
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
                    clearInterval(activePollInterval);
                    activePollInterval = null;
                    showError(data.reason || 'Pipeline failed');
                    // Show failure in report panel as well
                    document.getElementById('report-placeholder-text').textContent = `Pipeline Failed: ${data.reason}`;
                    document.getElementById('report-placeholder-text').style.color = 'var(--danger)';
                } else if (data.state === 'complete') {
                    clearInterval(activePollInterval);
                    activePollInterval = null;
                    progressBar.style.width = '100%';
                    processingText.textContent = 'Complete!';

                    setTimeout(() => loadDashboard(taskId), 500);
                }
            } catch (err) {
                clearInterval(activePollInterval);
                activePollInterval = null;
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
                tr.innerHTML = `<td>${escapeHtml(m.name)}</td><td>${escapeHtml(m.value)}</td><td>${escapeHtml(m.caveat)}</td>`;
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
                    arBtn.onclick = () => openArModal(`${API_BASE}/ar/${taskId}`);
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

    // --- Access key ---
    // The server may be configured with a shared key (KNEETWIN_API_KEY). Once
    // exchanged via /api/auth it sets an HttpOnly session cookie, which the
    // browser then attaches automatically to every request — including the
    // <img>, <model-viewer> and PDF-link loads that cannot carry a custom
    // header. So nothing else in this file needs to know about auth.
    const authModal = document.getElementById('auth-modal');
    const authForm = document.getElementById('auth-form');
    const authInput = document.getElementById('auth-key-input');
    const authError = document.getElementById('auth-error');

    function showAuthPrompt() {
        authModal.style.display = 'flex';
        authInput.focus();
    }

    authForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        authError.style.display = 'none';
        try {
            const res = await fetch(`${API_BASE}/auth`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ api_key: authInput.value }),
            });
            if (!res.ok) throw new Error('That key was not accepted.');
            authModal.style.display = 'none';
            // Reload so everything re-fetches with the session cookie, rather
            // than trying to replay whichever calls happened to fail first.
            window.location.reload();
        } catch (err) {
            authError.textContent = err.message;
            authError.style.display = 'block';
            authInput.select();
        }
    });

    async function startApp() {
        try {
            const res = await fetch(`${API_BASE}/auth/status`);
            const status = await res.json();
            if (status.auth_required && !status.authenticated) {
                showAuthPrompt();
                return;  // don't load anything until unlocked
            }
        } catch (e) {
            // Status check failing shouldn't block a local, unauthenticated
            // instance — carry on and let individual calls report their own
            // errors rather than showing a key prompt that may not apply.
            console.error('Could not determine auth status', e);
        }

        // ?task=<id> opens an already-processed case directly. Without it the
        // only route to the dashboard is uploading a scan, which is
        // impractical from a phone — and a phone is the only place AR runs.
        const requestedTask = new URLSearchParams(window.location.search).get('task');
        if (requestedTask) {
            uploadSection.style.display = 'none';
            dashboardSection.style.display = 'grid';
            if (window.initSliceViewer) initSliceViewer(requestedTask);
            loadDashboard(requestedTask);
        }
    }

    startApp();
});
