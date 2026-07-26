class KneeViewport {
    constructor(containerId, side, taskId, modality) {
        this.container = document.getElementById(containerId);
        this.side = side;
        this.taskId = taskId;
        this.modality = modality;
        this.container.innerHTML = '';
        
        this.scene = new THREE.Scene();
        this.scene.background = null;
        
        // Adjust near/far planes to prevent clipping/fading.
        // A far plane of 2000 is plenty for a single knee.
        this.camera = new THREE.PerspectiveCamera(45, this.container.clientWidth / this.container.clientHeight, 0.1, 2000);
        
        this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
        this.renderer.setSize(this.container.clientWidth, this.container.clientHeight);
        this.renderer.setPixelRatio(window.devicePixelRatio);
        this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
        this.renderer.toneMappingExposure = 1.0;
        this.container.appendChild(this.renderer.domElement);
        
        const ambientLight = new THREE.AmbientLight(0xffffff, 0.4);
        this.scene.add(ambientLight);
        const dirLight = new THREE.DirectionalLight(0xffffff, 0.5);
        dirLight.position.set(10, 20, 10);
        this.scene.add(dirLight);
        const backLight = new THREE.DirectionalLight(0xffffff, 0.15);
        backLight.position.set(-10, -20, -10);
        this.scene.add(backLight);
        
        this.controls = new THREE.OrbitControls(this.camera, this.renderer.domElement);
        this.controls.enableDamping = true;
        this.controls.dampingFactor = 0.05;
        
        this.kneeGroup = new THREE.Group();
        this.scene.add(this.kneeGroup);
        
        this.combinedBox = new THREE.Box3();
        this.animating = false;
    }
    
    startAnimation() {
        if (this.animating) return;
        this.animating = true;
        const animate = () => {
            if (!this.animating) return;
            requestAnimationFrame(animate);
            this.controls.update();
            this.renderer.render(this.scene, this.camera);
        };
        animate();
    }
    
    stopAnimation() {
        this.animating = false;
    }
    
    resize() {
        if (!this.container.clientWidth) return;
        this.camera.aspect = this.container.clientWidth / this.container.clientHeight;
        this.camera.updateProjectionMatrix();
        this.renderer.setSize(this.container.clientWidth, this.container.clientHeight);
        this.resetView(); // Ensure camera reframes correctly after a layout change
    }
    
    resetView() {
        if (this.combinedBox.isEmpty()) return;
        
        const center = new THREE.Vector3();
        this.combinedBox.getCenter(center);
        const size = new THREE.Vector3();
        this.combinedBox.getSize(size);
        
        const maxDim = Math.max(size.x, size.y, size.z);
        const fov = this.camera.fov * (Math.PI / 180);
        let cameraDist = Math.abs(maxDim / 2 / Math.tan(fov / 2));
        cameraDist *= 1.1; // Reduced from 1.5 so meshes fill more of the viewport
        
        // CT coordinate system: Y is Anterior (front), Z is Superior (up)
        // Set camera on the Anterior side looking at the center, with Superior as UP
        this.camera.position.set(center.x, center.y - cameraDist, center.z);
        this.camera.up.set(0, 0, 1);
        this.controls.target.copy(center);
        this.controls.update();
    }
    
    loadParts(parts) {
        const loader = new THREE.OBJLoader();
        let loadedCount = 0;
        
        parts.forEach(part => {
            const meshUrl = `http://localhost:8000/api/mesh/${this.taskId}/${part.file}`;
            loader.load(meshUrl, (obj) => {
                const isMRIBone = this.modality === 'MRI' && (part.file.includes('femur') || part.file.includes('tibia')) && !part.file.includes('cartilage');
                
                const material = new THREE.MeshStandardMaterial({
                    color: part.color,
                    roughness: 0.5,
                    metalness: 0.1,
                    side: THREE.DoubleSide,
                    transparent: isMRIBone,
                    opacity: isMRIBone ? 0.25 : 1.0
                });
                
                obj.traverse((child) => {
                    if (child.isMesh) {
                        child.material = material;
                        if (isMRIBone) {
                            child.userData.isMRIBone = true;
                        }
                    }
                });
                
                this.kneeGroup.add(obj);
                
                const box = new THREE.Box3().setFromObject(obj);
                if (this.combinedBox.isEmpty()) {
                    this.combinedBox.copy(box);
                } else {
                    this.combinedBox.union(box);
                }
                
                loadedCount++;
                if (loadedCount === parts.length) {
                    this.resetView();
                }
            });
        });
    }
    
    toggleBone() {
        this.kneeGroup.traverse((child) => {
            if (child.isMesh && child.userData.isMRIBone) {
                child.visible = !child.visible;
            }
        });
        this.renderer.render(this.scene, this.camera);
    }
}

// Global registry for resizing
window.activeViewports = [];

window.addEventListener("resize", () => {
    window.activeViewports.forEach(vp => vp.resize());
});

window.initViewer = async function(taskId) {
    const dashboardSection = document.getElementById("dashboard-section");
    const singlePanel = document.getElementById("single-viewer-panel");
    const legacyContainer = document.getElementById("canvas-container");
    const leftPanel = document.getElementById("left-knee-panel");
    const rightPanel = document.getElementById("right-knee-panel");
    
    // Stop old animations and clear
    window.activeViewports.forEach(vp => vp.stopAnimation());
    window.activeViewports = [];
    legacyContainer.innerHTML = '';
    document.getElementById("left-canvas-container").innerHTML = '';
    document.getElementById("right-canvas-container").innerHTML = '';
    
    // Remove old toggles
    document.querySelectorAll('.bone-toggle-btn').forEach(btn => btn.remove());
    
    try {
        const manifestRes = await fetch(`http://localhost:8000/api/manifest/${taskId}`);
        if (!manifestRes.ok) throw new Error("Failed to load manifest");
        const manifest = await manifestRes.json();
        
        const leftParts = manifest.parts.filter(p => p.side === 'left');
        const rightParts = manifest.parts.filter(p => p.side === 'right');
        
        if (leftParts.length > 0 && rightParts.length > 0) {
            // Bilateral
            dashboardSection.classList.add("bilateral");
            singlePanel.style.display = "none";
            leftPanel.style.display = "flex";
            rightPanel.style.display = "flex";
            
            const vpLeft = new KneeViewport("left-canvas-container", "left", taskId, manifest.modality);
            const vpRight = new KneeViewport("right-canvas-container", "right", taskId, manifest.modality);
            
            vpLeft.loadParts(leftParts);
            vpRight.loadParts(rightParts);
            
            vpLeft.startAnimation();
            vpRight.startAnimation();
            
            window.activeViewports.push(vpLeft, vpRight);
            
            document.getElementById("reset-left-btn").onclick = () => vpLeft.resetView();
            document.getElementById("reset-right-btn").onclick = () => vpRight.resetView();
            
            if (manifest.modality === 'MRI') {
                const btnLeft = document.createElement('button');
                btnLeft.className = 'btn-secondary btn-sm bone-toggle-btn';
                btnLeft.style.cssText = 'position: absolute; top: 1rem; left: 6rem; z-index: 10;';
                btnLeft.innerText = 'Toggle Bone';
                btnLeft.onclick = () => vpLeft.toggleBone();
                leftPanel.appendChild(btnLeft);
                
                const btnRight = document.createElement('button');
                btnRight.className = 'btn-secondary btn-sm bone-toggle-btn';
                btnRight.style.cssText = 'position: absolute; top: 1rem; left: 6rem; z-index: 10;';
                btnRight.innerText = 'Toggle Bone';
                btnRight.onclick = () => vpRight.toggleBone();
                rightPanel.appendChild(btnRight);
            }
            
        } else {
            // Unilateral
            dashboardSection.classList.remove("bilateral");
            singlePanel.style.display = "flex";
            leftPanel.style.display = "none";
            rightPanel.style.display = "none";
            
            const vp = new KneeViewport("canvas-container", "single", taskId, manifest.modality);
            vp.loadParts(manifest.parts);
            vp.startAnimation();
            
            if (manifest.modality === 'MRI') {
                const btn = document.createElement('button');
                btn.className = 'btn-secondary btn-sm bone-toggle-btn';
                btn.style.cssText = 'position: absolute; top: 1rem; left: 1rem; z-index: 10;';
                btn.innerText = 'Toggle Bone';
                btn.onclick = () => vp.toggleBone();
                singlePanel.appendChild(btn);
            }
            
            window.activeViewports.push(vp);
        }
        
        // Force a resize immediately to ensure dimensions are correct after CSS layout changes
        setTimeout(() => {
            window.activeViewports.forEach(vp => vp.resize());
        }, 50);
        
    } catch (err) {
        console.error("Error loading meshes:", err);
        legacyContainer.style.display = "block";
        legacyContainer.innerHTML = `<div style="color:red; padding: 2rem;">Failed to load 3D Viewer: ${err.message}</div>`;
    }
};
