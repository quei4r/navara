import { BoxGeometry, Mesh, PerspectiveCamera, Scene } from "three";
import { MeshBasicNodeMaterial, WebGPURenderer } from "three/webgpu";

const bootstrap = async () => {
  const canvas = document.createElement("canvas");
  canvas.style.width = "100%";
  canvas.style.height = "100%";
  const root = document.createElement("div");
  root.style.width = "100vw";
  root.style.height = "100vh";
  root.appendChild(canvas);
  document.body.appendChild(root);

  const renderer = new WebGPURenderer({
    canvas,
    antialias: true,
    stencil: true,
    logarithmicDepthBuffer: true,
  });
  await renderer.init();
  renderer.setSize(1280, 800, false);

  const scene = new Scene();
  const camera = new PerspectiveCamera(60, 1280 / 800, 1, 1000);
  camera.position.set(0, 0, 5);

  const mesh = new Mesh(
    new BoxGeometry(2, 2, 2),
    new MeshBasicNodeMaterial({ color: 0xff3333 }),
  );
  scene.add(mesh);

  renderer.setAnimationLoop(() => {
    mesh.rotation.x += 0.01;
    mesh.rotation.y += 0.02;
    renderer.setRenderTarget(null);
    renderer.render(scene, camera);
  });

  (window as unknown as Record<string, unknown>).__min = { renderer, mesh };
};

bootstrap().catch((e) => console.error(e));
