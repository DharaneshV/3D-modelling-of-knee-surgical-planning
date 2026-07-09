let currentTaskId = null;
let currentPlane = 'axial';
let volumeInfo = null;

// DOM Elements
const sliceImage = document.getElementById('slice-image');
const sliceSlider = document.getElementById('slice-slider');
const sliceLabel = document.getElementById('slice-label');
const wcSlider = document.getElementById('wc-slider');
const wwSlider = document.getElementById('ww-slider');
const wcLabel = document.getElementById('wc-label');
const wwLabel = document.getElementById('ww-label');
const planeBtns = document.querySelectorAll('.plane-btn');

function initSliceViewer(taskId) {
    currentTaskId = taskId;
    
    // Fetch volume info
    fetch(`/api/volume-info/${taskId}`)
        .then(res => {
            if (!res.ok) throw new Error("Volume info not found");
            return res.json();
        })
        .then(data => {
            volumeInfo = data;
            
            // Setup slider for initial plane
            updateSliderForPlane(currentPlane);
            
            // Auto-load mid-volume axial slice
            const midIndex = Math.floor(volumeInfo.num_slices[currentPlane] / 2);
            sliceSlider.value = midIndex;
            updateSliceImage();
        })
        .catch(err => console.error("Failed to init slice viewer:", err));
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
    
    // Get window values (0 means Auto on the backend)
    const wc = wcSlider.value;
    const ww = wwSlider.value;
    
    let url = `/api/slices/${currentTaskId}/${currentPlane}/${index}`;
    // If not default (0), append query params
    if (wc != 0) url += `?wc=${wc}&ww=${ww}`;
    else if (ww != 1000) url += `?ww=${ww}`; // Just in case
    
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

sliceSlider.addEventListener('input', () => {
    if (!volumeInfo) return;
    sliceLabel.textContent = `${sliceSlider.value}/${volumeInfo.num_slices[currentPlane] - 1}`;
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
    wcLabel.textContent = e.target.value == 0 ? "Auto" : e.target.value;
    debouncedUpdate();
});

wwSlider.addEventListener('input', (e) => {
    wwLabel.textContent = e.target.value == 1000 && wcSlider.value == 0 ? "Auto" : e.target.value;
    debouncedUpdate();
});
