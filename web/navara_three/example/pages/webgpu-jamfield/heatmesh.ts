// 地形贴合覆盖热图网格（Navara 版，替代 r185 的屏幕空间 deferred draping——
// Navara 前向路径拿不到场景颜色+深度，改用几何方案）：
// 一张 5500×5500m 的网格平面，顶点高度取三层障碍场列数据中最精可用值
// （high>mid>low 覆盖优先级，低档全盒兜底）+2m 抬升防 z-fight；
// fragment 逐像素把局部坐标变换进三层盒 uvw 采样各自 display 纹理
// （alpha×300=score，跨层取 max——score 同为 dB 量纲，跨层 max 严格等价
// 全局选段），工程色带同 r185 heatmap.ts（深紫→蓝→青→绿→黄→橙→红）。
// 收敛前 display 纹理会 ping-pong 换对象，refreshTextures() 每帧同步
// TextureNode.value（NodeSampledTexture.update() 检测 value 变更自动重绑）。
import {
  vec3,
  float,
  uniform,
  texture3D,
  mix,
  clamp,
  smoothstep,
  max,
  varying,
  positionGeometry,
} from "three/tsl";
import * as THREE from "three/webgpu";

import type { JamField } from "./jamfield";
import type { ObstacleField } from "./obstacles";

export type HeatTier = {
  jam: JamField;
  obstacle: ObstacleField;
  boxMin: THREE.Vector3;
  boxSize: THREE.Vector3;
};

export type HeatMesh = {
  mesh: THREE.Mesh;
  /** 每帧同步 synth ping-pong 后的最新场纹理（收敛前纹理会换） */
  refreshTextures: () => void;
  /** 显示总透明度（0..1，乘进 opacityNode，不触发重算） */
  setOpacity: (v: number) => void;
  dispose: () => void;
};

export function createHeatMesh(
  tiers: HeatTier[],
  opts?: { size?: number; segments?: number; lift?: number },
): HeatMesh {
  const SIZE = opts?.size ?? 5500;
  const SEG = opts?.segments ?? 255;
  // 抬升防 z-fight：Navara 地形（mapterhorn DEM，城区只有粗粒度覆盖）与
  // 障碍场采样的 AWS Terrarium 基准有米级差异，2m 抬升会被局部地形盖过
  // （热图出现建筑形破洞，已实测），8m 兜住该误差
  const LIFT = opts?.lift ?? 8;

  // 顶点高度：按盒面积升序（最精优先）找第一个覆盖该点的层
  const ordered = [...tiers].sort(
    (a, b) => a.boxSize.x * a.boxSize.z - b.boxSize.x * b.boxSize.z,
  );
  const heightAt = (x: number, z: number): number => {
    for (const t of ordered) {
      if (
        Math.abs(x - (t.boxMin.x + t.boxSize.x / 2)) <= t.boxSize.x / 2 &&
        Math.abs(z - (t.boxMin.z + t.boxSize.z / 2)) <= t.boxSize.z / 2
      ) {
        return t.obstacle.surfaceY(x, z);
      }
    }
    return 0;
  };

  // PlaneGeometry 转平到 XZ：rotateX(-π/2) 后 (x, y, 0)→(x, 0, -y)，
  // 平面 +y 边朝北（ENU z 南 ⇒ 北=-z），与障碍场列布局一致
  const geom = new THREE.PlaneGeometry(SIZE, SIZE, SEG, SEG);
  geom.rotateX(-Math.PI / 2);
  const pos = geom.getAttribute("position") as THREE.BufferAttribute;
  for (let i = 0; i < pos.count; i++) {
    pos.setY(i, heightAt(pos.getX(i), pos.getZ(i)) + LIFT);
  }
  pos.needsUpdate = true;
  geom.computeVertexNormals();
  geom.computeBoundingSphere();

  // 工程色带：t = score/30 ∈[0,1]，深紫(刚压制)→蓝→青→绿→黄→橙→红(饱和)
  // （纯表达式嵌套，无 toVar/assign——assign 必须在 Fn() 栈内，已踩坑）
  const heatColor = (t: any) => {
    const c0 = mix(
      vec3(0.1, 0.05, 0.45),
      vec3(0.1, 0.45, 0.95),
      smoothstep(0.0, 0.18, t),
    );
    const c1 = mix(c0, vec3(0.1, 0.85, 0.85), smoothstep(0.18, 0.38, t));
    const c2 = mix(c1, vec3(0.35, 0.9, 0.25), smoothstep(0.38, 0.58, t));
    const c3 = mix(c2, vec3(0.98, 0.9, 0.15), smoothstep(0.58, 0.78, t));
    const c4 = mix(c3, vec3(0.98, 0.45, 0.1), smoothstep(0.78, 0.9, t));
    return mix(c4, vec3(0.9, 0.05, 0.05), smoothstep(0.9, 1.0, t));
  };

  // 顶点局部坐标（含烘焙高度）传给 fragment，逐像素变换进各层盒 uvw 采样
  const vLocal = varying(positionGeometry);
  const texNodes: { value: THREE.Texture }[] = [];
  let bestScore: any = float(0);
  for (const t of tiers) {
    // boxMin/boxSize 烘焙为常量节点（源不移动，无需 uniform）
    const uvw = vLocal.sub(vec3(t.boxMin)).div(vec3(t.boxSize));
    const inBox = uvw.x
      .greaterThanEqual(0)
      .and(uvw.y.greaterThanEqual(0))
      .and(uvw.z.greaterThanEqual(0))
      .and(uvw.x.lessThanEqual(1))
      .and(uvw.y.lessThanEqual(1))
      .and(uvw.z.lessThanEqual(1));
    const texNode = texture3D(t.jam.getDisplayTexture(), uvw, float(0));
    texNodes.push(texNode as unknown as { value: THREE.Texture });
    bestScore = max(bestScore, texNode.a.mul(300).mul(inBox.toFloat()));
  }

  const uOpacity = uniform(1.0); // 面板总透明度（UI 直调，不走 synth）
  const material = new THREE.MeshBasicNodeMaterial();
  material.colorNode = heatColor(clamp(bestScore.div(30), 0, 1));
  // 显示强度：score 0→6dB 淡入，上限 0.55（不盖死卫星底图）× 源透明度
  material.opacityNode = smoothstep(float(0), float(6), bestScore)
    .mul(0.55)
    .mul(uOpacity);
  material.transparent = true;
  material.depthWrite = false; // 贴地网格不写深，避免与壳/地形互挡

  const mesh = new THREE.Mesh(geom, material);
  mesh.name = "jam-heatmap";

  return {
    mesh,
    refreshTextures: () => {
      for (let i = 0; i < texNodes.length && i < tiers.length; i++) {
        texNodes[i].value = tiers[i].jam.getDisplayTexture();
      }
    },
    setOpacity: (v: number) => {
      uOpacity.value = v;
    },
    dispose: () => {
      geom.dispose();
      material.dispose();
    },
  };
}
