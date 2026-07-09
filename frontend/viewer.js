window.initViewer = async function(taskId) {
    const container = document.getElementById('canvas-container');
    container.innerHTML = ''; // Clear existing

    // 1. Setup Scene, Camera, Renderer
    const scene = new THREE.Scene();
    scene.background = null; // transparent to show glassmorphism behind

    const camera = new THREE.PerspectiveCamera(45, container.clientWidth / container.clientHeight, 0.1, 1000);
    
    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setSize(container.clientWidth, container.clientHeight);
    renderer.setPixelRatio(window.devicePixelRatio);
    container.appendChild(renderer.domElement);

    // 2. Lighting
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
    scene.add(ambientLight);

    const dirLight = new THREE.DirectionalLight(0xffffff, 0.8);
    dirLight.position.set(10, 20, 10);
    scene.add(dirLight);
    
    const backLight = new THREE.DirectionalLight(0xffffff, 0.3);
    backLight.position.set(-10, -20, -10);
    scene.add(backLight);

    // 3. Orbit Controls
    const controls = new THREE.OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;

    // 4. Load Manifest and Meshes
    const kneeGroup = new THREE.Group();
    scene.add(kneeGroup);

    try {
        const manifestRes = await fetch(`http://localhost:8000/api/manifest/${taskId}`);
        if (!manifestRes.ok) throw new Error("Failed to load manifest");
        const manifest = await manifestRes.json();

        const loader = new THREE.OBJLoader();
        
        // Bounding box for the whole group to center the camera
        const combinedBox = new THREE.Box3();

        for (const part of manifest.parts) {
            const meshUrl = `http://localhost:8000/api/mesh/${taskId}/${part.file}`;
            
            // Load OBJ
            loader.load(meshUrl, (obj) => {
                // Apply Material
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
                
                kneeGroup.add(obj);
                
                // Expand bounding box to include this part
                const box = new THREE.Box3().setFromObject(obj);
                if (combinedBox.isEmpty()) {
                    combinedBox.copy(box);
                } else {
                    combinedBox.union(box);
                }
                
                // Recenter camera
                const center = new THREE.Vector3();
                combinedBox.getCenter(center);
                const size = new THREE.Vector3();
                combinedBox.getSize(size);
                
                const maxDim = Math.max(size.x, size.y, size.z);
                const fov = camera.fov * (Math.PI / 180);
                let cameraZ = Math.abs(maxDim / 2 / Math.tan(fov / 2));
                cameraZ *= 1.5; // Zoom out a bit
                
                camera.position.set(center.x, center.y, center.z + cameraZ);
                controls.target.copy(center);
                controls.update();
            });
        }
    } catch (err) {
        console.error("Error loading meshes:", err);
        container.innerHTML = `<div style="color:red; padding: 2rem;">Failed to load 3D Viewer: ${err.message}</div>`;
    }

    // 5. Animation Loop
    function animate() {
        requestAnimationFrame(animate);
        controls.update();
        renderer.render(scene, camera);
    }
    animate();

    // 6. Handle Resize
    window.addEventListener('resize', () => {
        if (!container.clientWidth) return;
        camera.aspect = container.clientWidth / container.clientHeight;
        camera.updateProjectionMatrix();
        renderer.setSize(container.clientWidth, container.clientHeight);
    });
};
