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
        // part.file -> loaded Object3D, so individual structures can be hidden
        // (needed to expose bone surfaces for implant placement).
        this.partObjects = {};
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
            const meshUrl = `/api/mesh/${this.taskId}/${part.file}`;
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
                    }
                });
                
                this.kneeGroup.add(obj);
                this.partObjects[part.file] = obj;

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
    
    setPartVisible(file, visible) {
        const obj = this.partObjects[file];
        if (!obj) return;
        obj.visible = visible;
        this.renderer.render(this.scene, this.camera);
    }

    /**
     * MRI bone is drawn semi-transparent so cartilage reads through it. That
     * hides the resection cut, so bone is made solid while a cut is displayed.
     */
    setBoneOpaque(opaque) {
        ['femur_unknown.obj', 'tibia_unknown.obj'].forEach(file => {
            const obj = this.partObjects[file];
            if (!obj) return;
            obj.traverse((c) => {
                if (!c.isMesh) return;
                c.material.transparent = !opaque;
                c.material.opacity = opaque ? 1.0 : 0.25;
                c.material.needsUpdate = true;
            });
        });
        this.renderer.render(this.scene, this.camera);
    }

    /**
     * Add a mesh that is not in the manifest — the implant, which is generated
     * on demand rather than by the segmentation pipeline. Passing null for
     * colour removes it again.
     */
    setExtraPart(file, color) {
        const existing = this.partObjects[file];
        if (!color) {
            if (existing) {
                this.kneeGroup.remove(existing);
                delete this.partObjects[file];
                this.renderer.render(this.scene, this.camera);
            }
            return;
        }
        if (existing) return;

        new THREE.OBJLoader().load(
            `/api/mesh/${this.taskId}/${file}`,
            (obj) => {
                // Low metalness on purpose. MeshStandardMaterial metals have no
                // diffuse term, so without an environment map to reflect a highly
                // metallic surface renders black under this scene's ambient +
                // directional lights. Keep it dielectric and lean on roughness
                // for the polished look instead.
                const material = new THREE.MeshStandardMaterial({
                    color: color, roughness: 0.35, metalness: 0.15,
                    side: THREE.DoubleSide,
                });
                obj.traverse((c) => { if (c.isMesh) c.material = material; });
                this.kneeGroup.add(obj);
                this.partObjects[file] = obj;
                this.renderer.render(this.scene, this.camera);
            });
    }

    /**
     * Swap a loaded part for a different mesh file, reusing its material so the
     * resected bone keeps the colour and transparency of the intact one.
     * Passing null for replacementFile restores the original.
     */
    swapPart(file, replacementFile) {
        const existing = this.partObjects[file];
        if (!existing) return;

        const target = replacementFile || file;
        if (existing.userData.showing === target) return;

        const loader = new THREE.OBJLoader();
        loader.load(`/api/mesh/${this.taskId}/${target}`, (obj) => {
            let material = null;
            existing.traverse((c) => { if (c.isMesh && !material) material = c.material; });

            obj.traverse((c) => { if (c.isMesh && material) c.material = material; });
            obj.visible = existing.visible;
            obj.userData.showing = target;

            this.kneeGroup.remove(existing);
            this.kneeGroup.add(obj);
            this.partObjects[file] = obj;
            this.renderer.render(this.scene, this.camera);
        });
    }
}

// Per-part show/hide panel. Replaces the old all-or-nothing "Toggle Bone"
// button so individual structures can be removed from the scene.
function buildPartControls(viewport, parts, panel) {
    const box = document.createElement('div');
    box.className = 'part-controls';
    box.style.cssText = 'position: absolute; top: 3.5rem; left: 1rem; z-index: 10;' +
        'background: rgba(0,0,0,0.55); border-radius: 8px; padding: 0.5rem 0.75rem;' +
        'font-size: 0.75rem; line-height: 1.6; backdrop-filter: blur(4px);';

    parts.forEach(part => {
        const id = `part-${viewport.side}-${part.file.replace(/\W/g, '')}`;
        const row = document.createElement('label');
        row.style.cssText = 'display: flex; align-items: center; gap: 0.4rem; cursor: pointer;';

        const cb = document.createElement('input');
        cb.type = 'checkbox';
        cb.checked = true;
        cb.id = id;
        cb.style.cursor = 'pointer';
        cb.addEventListener('change', () => viewport.setPartVisible(part.file, cb.checked));

        const swatch = document.createElement('span');
        swatch.style.cssText = `width: 10px; height: 10px; border-radius: 2px;` +
            `background: ${part.color}; display: inline-block; flex-shrink: 0;`;

        const text = document.createElement('span');
        text.textContent = part.label;

        row.append(cb, swatch, text);
        box.appendChild(row);
    });

    panel.appendChild(box);
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
    
    // Remove old part-visibility panels
    document.querySelectorAll('.part-controls').forEach(el => el.remove());
    
    try {
        const manifestRes = await fetch(`/api/manifest/${taskId}`);
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
            
            buildPartControls(vpLeft, leftParts, leftPanel);
            buildPartControls(vpRight, rightParts, rightPanel);


        } else {
            // Unilateral
            dashboardSection.classList.remove("bilateral");
            singlePanel.style.display = "flex";
            leftPanel.style.display = "none";
            rightPanel.style.display = "none";
            
            const vp = new KneeViewport("canvas-container", "single", taskId, manifest.modality);
            vp.loadParts(manifest.parts);
            vp.startAnimation();
            
            buildPartControls(vp, manifest.parts, singlePanel);

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
