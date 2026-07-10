class KneeViewport {
    constructor(containerId, side, taskId) {
        this.container = document.getElementById(containerId);
        this.side = side;
        this.taskId = taskId;
        this.container.innerHTML = '';
        
        this.scene = new THREE.Scene();
        this.scene.background = null;
        
        // Adjust near/far planes to prevent clipping/fading.
        // A far plane of 2000 is plenty for a single knee.
        this.camera = new THREE.PerspectiveCamera(45, this.container.clientWidth / this.container.clientHeight, 0.1, 2000);
        
        this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
        this.renderer.setSize(this.container.clientWidth, this.container.clientHeight);
        this.renderer.setPixelRatio(window.devicePixelRatio);
        this.container.appendChild(this.renderer.domElement);
        
        const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
        this.scene.add(ambientLight);
        const dirLight = new THREE.DirectionalLight(0xffffff, 0.8);
        dirLight.position.set(10, 20, 10);
        this.scene.add(dirLight);
        const backLight = new THREE.DirectionalLight(0xffffff, 0.3);
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
        let cameraZ = Math.abs(maxDim / 2 / Math.tan(fov / 2));
        cameraZ *= 1.1; // Reduced from 1.5 so meshes fill more of the viewport
        
        this.camera.position.set(center.x, center.y, center.z + cameraZ);
        this.controls.target.copy(center);
        this.controls.update();
    }
    
    loadParts(parts) {
        const loader = new THREE.OBJLoader();
        let loadedCount = 0;
        
        parts.forEach(part => {
            const meshUrl = `http://localhost:8000/api/mesh/${this.taskId}/${part.file}`;
            loader.load(meshUrl, (obj) => {
                const material = new THREE.MeshStandardMaterial({
                    color: part.color,
                    roughness: 0.5,
                    metalness: 0.1,
                    side: THREE.DoubleSide
                });
                
                obj.traverse((child) => {
                    if (child.isMesh) {
                        child.material = material;
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
            
            const vpLeft = new KneeViewport("left-canvas-container", "left", taskId);
            const vpRight = new KneeViewport("right-canvas-container", "right", taskId);
            
            vpLeft.loadParts(leftParts);
            vpRight.loadParts(rightParts);
            
            vpLeft.startAnimation();
            vpRight.startAnimation();
            
            window.activeViewports.push(vpLeft, vpRight);
            
            document.getElementById("reset-left-btn").onclick = () => vpLeft.resetView();
            document.getElementById("reset-right-btn").onclick = () => vpRight.resetView();
            
        } else {
            // Unilateral
            dashboardSection.classList.remove("bilateral");
            singlePanel.style.display = "flex";
            leftPanel.style.display = "none";
            rightPanel.style.display = "none";
            
            const vp = new KneeViewport("canvas-container", "single", taskId);
            vp.loadParts(manifest.parts);
            vp.startAnimation();
            
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
