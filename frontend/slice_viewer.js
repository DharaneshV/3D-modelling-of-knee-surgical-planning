let currentTaskId = null;
let currentPlane = 'axial';
let volumeInfo = null;
// Explicit flag rather than a sentinel slider value — 0 is a legitimate MRI
// intensity, so "wc === 0 means auto" silently broke manual windowing on MR.
let autoWindow = true;

// DOM Elements
const sliceImage = document.getElementById('slice-image');
const sliceSlider = document.getElementById('slice-slider');
const sliceLabel = document.getElementById('slice-label');
const wcSlider = document.getElementById('wc-slider');
const wwSlider = document.getElementById('ww-slider');
const wcLabel = document.getElementById('wc-label');
const wwLabel = document.getElementById('ww-label');
const planeBtns = document.querySelectorAll('.plane-btn');

// Shared with script.js, which loads after this file and references the same
// global rather than redeclaring it — classic (non-module) scripts on one
// page share a top-level scope, so a single const here is visible there too.
const API_BASE = '/api';

function initSliceViewer(taskId) {
    currentTaskId = taskId;
    
    // Fetch volume info
    fetch(`${API_BASE}/volume-info/${taskId}`)
        .then(res => {
            if (!res.ok) throw new Error("Volume info not found");
            return res.json();
        })
        .then(data => {
            volumeInfo = data;

            // Setup slider for initial plane
            updateSliderForPlane(currentPlane);
            configureWindowSliders();

            // Auto-load mid-volume axial slice
            const midIndex = Math.floor(volumeInfo.num_slices[currentPlane] / 2);
            sliceSlider.value = midIndex;
            updateSliceImage();
        })
        .catch(err => console.error("Failed to init slice viewer:", err));
}

function configureWindowSliders() {
    // Scale the window/level controls to the volume's real intensity range.
    // CT is in Hounsfield units, but MRI has no standardised scale, so fixed
    // HU bounds leave almost the entire slider travel doing nothing on MR.
    if (!volumeInfo || !volumeInfo.intensity) return;

    const { min, max, p1, p99 } = volumeInfo.intensity;
    const span = Math.max(p99 - p1, 1);
    const step = Math.max(Math.round(span / 200), 1);

    wcSlider.min = Math.floor(min);
    wcSlider.max = Math.ceil(max);
    wcSlider.step = step;
    wcSlider.value = Math.round(p1 + span / 2);

    wwSlider.min = step;
    wwSlider.max = Math.ceil(span * 2);
    wwSlider.step = step;
    wwSlider.value = Math.round(span);

    autoWindow = true;
    wcLabel.textContent = 'Auto';
    wwLabel.textContent = 'Auto';
}

function updateSliderForPlane(plane) {
    if (!volumeInfo) return;
    const maxSlices = volumeInfo.num_slices[plane];
    sliceSlider.max = maxSlices - 1;
    
    // Reset to middle
    const midIndex = Math.floor(maxSlices / 2);
    sliceSlider.value = midIndex;
    sliceLabel.textContent = `${midIndex}/${maxSlices - 1}`;
}

function updateSliceImage() {
    if (!currentTaskId || !volumeInfo) return;
    
    const index = sliceSlider.value;
    sliceLabel.textContent = `${index}/${volumeInfo.num_slices[currentPlane] - 1}`;
    
    // Omitting wc/ww entirely tells the backend to auto-window this slice.
    let url = `${API_BASE}/slices/${currentTaskId}/${currentPlane}/${index}`;
    if (!autoWindow) {
        url += `?wc=${wcSlider.value}&ww=${wwSlider.value}`;
    }

    sliceImage.onerror = () => {
        sliceImage.onerror = null;  // avoid a loop if the placeholder itself 404s
        console.error(`Failed to load slice ${currentPlane}/${index}`);
    };
    sliceImage.src = url;
}

// Event Listeners
planeBtns.forEach(btn => {
    btn.addEventListener('click', (e) => {
        planeBtns.forEach(b => b.classList.remove('active'));
        e.target.classList.add('active');
        currentPlane = e.target.dataset.plane;
        updateSliderForPlane(currentPlane);
        updateSliceImage();
    });
});

let sliceDebounceTimeout;
sliceSlider.addEventListener('input', () => {
    if (!volumeInfo) return;
    sliceLabel.textContent = `${sliceSlider.value}/${volumeInfo.num_slices[currentPlane] - 1}`;
    
    clearTimeout(sliceDebounceTimeout);
    sliceDebounceTimeout = setTimeout(() => {
        updateSliceImage();
    }, 30);
});

sliceSlider.addEventListener('change', updateSliceImage);

let debounceTimeout;
function debouncedUpdate() {
    clearTimeout(debounceTimeout);
    debounceTimeout = setTimeout(() => {
        updateSliceImage();
    }, 150);
}

wcSlider.addEventListener('input', (e) => {
    autoWindow = false;
    wcLabel.textContent = e.target.value;
    wwLabel.textContent = wwSlider.value;
    debouncedUpdate();
});

wwSlider.addEventListener('input', (e) => {
    autoWindow = false;
    wwLabel.textContent = e.target.value;
    wcLabel.textContent = wcSlider.value;
    debouncedUpdate();
});

const autoWindowBtn = document.getElementById('auto-window-btn');
if (autoWindowBtn) {
    autoWindowBtn.addEventListener('click', () => {
        configureWindowSliders();
        updateSliceImage();
    });
}
