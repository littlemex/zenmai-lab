// The three.js view of a trace: the token sequence in front, the Engram table of one layer as a
// wall of columns behind it, and the rows one position reads travelling between the two.

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { CSS2DObject, CSS2DRenderer } from 'three/addons/renderers/CSS2DRenderer.js';
import { ngramColor, rankColor } from './colors.js';
import { ngramPositions, ownerOf, tokenLabel } from './trace.js';

const TILE = { width: 1.5, gap: 0.25, height: 0.5, depth: 0.25 };
const STEP = TILE.width + TILE.gap;
const WALL = { radius: 10.5, base: 1.4, height: 7.2, span: Math.PI * 0.6, z: -1.5, maxSlab: 0.34, groupGap: 1.4 };
const ET = { y: 1.15, width: 1.4, height: 0.24 };
// tokens drawn on each side of the current one; the rest are hidden so long traces stay cheap
const TOKEN_WINDOW = 24;
const CURVE_SEGMENTS = 24;

function label(text, className) {
  const element = document.createElement('div');
  element.className = className;
  element.textContent = text;
  return new CSS2DObject(element);
}

function bezier(a, control, b, s, out) {
  const u = 1 - s;
  return out.set(
    u * u * a.x + 2 * u * s * control.x + s * s * b.x,
    u * u * a.y + 2 * u * s * control.y + s * s * b.y,
    u * u * a.z + 2 * u * s * control.z + s * s * b.z,
  );
}

const smooth = (x) => x * x * (3 - 2 * x);
const clamp01 = (x) => Math.min(1, Math.max(0, x));

function dispose(group) {
  const materials = new Set();
  group.traverse((object) => {
    if (object.isCSS2DObject) object.element.remove();
    object.geometry?.dispose();
    for (const material of [].concat(object.material ?? [])) materials.add(material);
    if (object.isInstancedMesh) object.dispose();
  });
  materials.forEach((material) => material.dispose());
  group.clear();
}

export class EngramScene {
  constructor(container) {
    this.container = container;
    this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    container.appendChild(this.renderer.domElement);
    this.labels = new CSS2DRenderer();
    this.labels.domElement.className = 'label-layer';
    container.appendChild(this.labels.domElement);

    this.scene = new THREE.Scene();
    this.scene.fog = new THREE.Fog(0x05070a, 18, 42);
    this.camera = new THREE.PerspectiveCamera(42, 1, 0.1, 200);
    this.camera.position.set(0, 5.4, 16);
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.target.set(0, 2.5, -2);
    this.controls.maxPolarAngle = Math.PI * 0.55;

    this.scene.add(new THREE.AmbientLight(0xffffff, 0.55));
    const key = new THREE.DirectionalLight(0xffffff, 1.1);
    key.position.set(4, 10, 8);
    this.scene.add(key);
    const floor = new THREE.GridHelper(400, 400, 0x1b2a22, 0x0e1612);
    floor.position.y = -1.6;
    this.scene.add(floor);

    this.root = new THREE.Group();
    this.wall = new THREE.Group();
    this.scene.add(this.root, this.wall);

    this.position = 0;
    this.followX = 0;
    this.animation = null;
    this.linksDirty = false;
    this.speed = 1;
    this.onStepEnd = null;
    new ResizeObserver(() => this.resize()).observe(container);
    this.resize();
    this.renderer.setAnimationLoop(() => this.frame());
  }

  resize() {
    const { clientWidth: w, clientHeight: h } = this.container;
    if (!w || !h) return;
    this.renderer.setSize(w, h);
    this.labels.setSize(w, h);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
  }

  // trace and index must have passed checkTrace: every layer has the same columns
  setTrace(trace, index) {
    this.trace = trace;
    this.index = index;
    this.position = 0;
    this.animation = null;
    this.flight = null;
    dispose(this.root);
    dispose(this.wall);
    this.markers = null;
    this.buildTokens();
    this.buildFlight();
  }

  setView(layerIndex, worldSize) {
    this.layerIndex = layerIndex;
    this.worldSize = worldSize;
    dispose(this.wall);
    this.buildWall();
    this.buildMarkers();
    this.show(this.position, false);
  }

  tileX(t) {
    return t * STEP;
  }

  buildTokens() {
    const { tokens, model } = this.trace;
    const geometry = new THREE.BoxGeometry(TILE.width, TILE.height, TILE.depth);
    const padMaterial = new THREE.MeshStandardMaterial({ color: 0x10151a, transparent: true, opacity: 0.45 });
    this.pads = [];
    for (let t = -(model.max_ngram_size - 1); t < 0; t += 1) {
      const tile = new THREE.Mesh(geometry, padMaterial);
      tile.position.set(this.tileX(t), 0, 3);
      tile.add(label('pad', 'token-label pad'));
      this.root.add(tile);
      this.pads.push(tile);
    }
    this.tiles = tokens.map((token, t) => {
      const tile = new THREE.Mesh(geometry, new THREE.MeshStandardMaterial({ color: 0x1c232b }));
      tile.position.set(this.tileX(t), 0, 3);
      tile.add(label(tokenLabel(token), 'token-label'));
      this.root.add(tile);
      return tile;
    });
    this.brackets = [];
    for (let n = 2; n <= model.max_ngram_size; n += 1) {
      const bar = new THREE.Mesh(
        new THREE.BoxGeometry(1, 0.07, 0.07),
        new THREE.MeshBasicMaterial({ color: ngramColor(n), transparent: true, opacity: 0.95 }),
      );
      const tag = label(`${n}-gram`, 'bracket-label');
      tag.element.style.color = ngramColor(n);
      tag.center.set(1.08, 0.5);
      tag.position.set(-0.5, 0, 0);
      bar.add(tag);
      bar.userData.n = n;
      this.root.add(bar);
      this.brackets.push(bar);
    }
  }

  columnLayout() {
    const columns = this.trace.layers[this.layerIndex].columns;
    const slots = [];
    let slot = 0;
    columns.forEach((column, c) => {
      if (c > 0 && column.n !== columns[c - 1].n) slot += WALL.groupGap;
      slots.push(slot);
      slot += 1;
    });
    const last = slots[slots.length - 1];
    const spacing = last > 0 ? (WALL.radius * WALL.span) / last : WALL.radius * WALL.span;
    this.slab = Math.min(WALL.maxSlab, spacing * 0.55);
    return columns.map((column, c) => {
      const angle = last > 0 ? -WALL.span / 2 + (WALL.span * slots[c]) / last : 0;
      return { column, angle, x: WALL.radius * Math.sin(angle), z: WALL.z - WALL.radius * Math.cos(angle) + WALL.radius * 0.35 };
    });
  }

  buildWall() {
    const layer = this.trace.layers[this.layerIndex];
    const part = Math.ceil(layer.num_rows / this.worldSize);
    const layout = this.columnLayout();
    const slab = this.slab;
    const groupHeads = new Map();
    this.pivots = layout.map(({ column, angle, x, z }) => {
      const pivot = new THREE.Object3D();
      pivot.position.set(x, WALL.base, z);
      pivot.rotation.y = -angle;
      const color = new THREE.Color(ngramColor(column.n));
      const body = new THREE.Mesh(
        new THREE.BoxGeometry(slab, WALL.height, 0.05),
        new THREE.MeshStandardMaterial({ color: 0x0d1318, emissive: color.clone().multiplyScalar(0.07), transparent: true, opacity: 0.9 }),
      );
      body.position.y = WALL.height / 2;
      body.add(new THREE.LineSegments(
        new THREE.EdgesGeometry(body.geometry),
        new THREE.LineBasicMaterial({ color: color.clone().multiplyScalar(0.3) }),
      ));
      pivot.add(body);

      // the ranks that hold this column's rows, as a strip down its left edge
      let start = column.offset;
      const end = column.offset + column.size;
      while (start < end) {
        const owner = ownerOf(start, layer.num_rows, this.worldSize);
        const stop = Math.min(end, (owner + 1) * part);
        const h = ((stop - start) / column.size) * WALL.height;
        const band = new THREE.Mesh(
          new THREE.BoxGeometry(0.1, h, 0.08),
          new THREE.MeshBasicMaterial({ color: rankColor(owner, this.worldSize) }),
        );
        band.position.set(-slab / 2 - 0.09, ((start - column.offset) / column.size) * WALL.height + h / 2, 0);
        pivot.add(band);
        start = stop;
      }
      if (!groupHeads.has(column.n)) groupHeads.set(column.n, []);
      groupHeads.get(column.n).push(pivot);
      this.wall.add(pivot);
      pivot.updateMatrix();
      return pivot;
    });
    for (const [n, pivots] of groupHeads) {
      const middle = pivots[Math.floor(pivots.length / 2)];
      const tag = label(`${n}-gram`, 'wall-label');
      tag.element.style.color = ngramColor(n);
      tag.position.set(middle.position.x, WALL.base + WALL.height + 0.45, middle.position.z);
      this.wall.add(tag);
    }
    const title = label(
      `Layer ${layer.layer_id} table · ${layer.num_rows.toLocaleString('en-US')} rows · ${layer.columns.length} columns`,
      'wall-title',
    );
    title.position.set(0, WALL.base + WALL.height + 1.3, WALL.z - WALL.radius * 0.62);
    this.wall.add(title);
  }

  // where on the wall (in wall-group coordinates) a row sits
  rowPoint(c, row, out = new THREE.Vector3()) {
    const column = this.trace.layers[this.layerIndex].columns[c];
    const y = ((row - column.offset + 0.5) / column.size) * WALL.height;
    return out.set(0, y, 0.06).applyMatrix4(this.pivots[c].matrix);
  }

  // One instance per lookup, in position order, so showing positions 0..t is drawing a prefix.
  buildMarkers() {
    const columns = this.trace.layers[this.layerIndex].columns;
    const lookups = this.index[this.layerIndex].lookups;
    this.markers = new THREE.InstancedMesh(
      new THREE.BoxGeometry(this.slab * 1.35, 0.055, 0.04),
      new THREE.MeshBasicMaterial({ transparent: true, opacity: 0.5, blending: THREE.AdditiveBlending, depthWrite: false }),
      lookups.length * columns.length,
    );
    this.markers.frustumCulled = false;
    const point = new THREE.Vector3();
    const matrix = new THREE.Matrix4();
    const colors = columns.map((column) => new THREE.Color(ngramColor(column.n)));
    lookups.forEach((perPos, t) =>
      perPos.forEach(({ row }, c) => {
        const i = t * columns.length + c;
        this.rowPoint(c, row, point);
        this.markers.setMatrixAt(i, matrix.makeRotationY(this.pivots[c].rotation.y).setPosition(point));
        this.markers.setColorAt(i, colors[c]);
      }),
    );
    this.wall.add(this.markers);

    this.current = columns.map((_, c) => {
      const mesh = new THREE.Mesh(
        new THREE.BoxGeometry(this.slab * 2.2, 0.09, 0.05),
        new THREE.MeshBasicMaterial({ color: 0xffffff, transparent: true, blending: THREE.AdditiveBlending }),
      );
      mesh.rotation.y = this.pivots[c].rotation.y;
      this.wall.add(mesh);
      return mesh;
    });
  }

  buildFlight() {
    const columns = this.trace.layers[0].columns.length;
    const sphere = new THREE.SphereGeometry(0.075, 12, 12);
    this.particles = Array.from({ length: columns }, () => {
      const mesh = new THREE.Mesh(sphere, new THREE.MeshBasicMaterial({ transparent: true, blending: THREE.AdditiveBlending }));
      mesh.visible = false;
      this.root.add(mesh);
      return mesh;
    });
    this.et = new THREE.Group();
    const w = ET.width / columns;
    this.etCells = Array.from({ length: columns }, (_, c) => {
      const cell = new THREE.Mesh(new THREE.BoxGeometry(w * 0.86, ET.height, 0.1), new THREE.MeshBasicMaterial({ transparent: true }));
      cell.position.set(-ET.width / 2 + w * (c + 0.5), 0, 0);
      this.et.add(cell);
      return cell;
    });
    const tag = label('e_t', 'et-label');
    tag.position.set(-ET.width / 2 - 0.25, 0, 0);
    this.et.add(tag);
    this.gateBars = [];
    if (this.trace.gates) {
      for (let k = 0; k < this.trace.model.hc_mult; k += 1) {
        const bar = new THREE.Mesh(new THREE.BoxGeometry(0.09, 1, 0.09), new THREE.MeshBasicMaterial({ color: 0xfbd332 }));
        bar.position.set(ET.width / 2 + 0.2 + k * 0.14, 0, 0);
        this.et.add(bar);
        this.gateBars.push(bar);
      }
    }
    this.root.add(this.et);

    // the lines from the current token to its rows, and the arcs from the earlier token that
    // read the same row first
    const lineGeometry = () => new THREE.BufferGeometry().setAttribute(
      'position', new THREE.BufferAttribute(new Float32Array((CURVE_SEGMENTS + 1) * 3), 3));
    this.links = Array.from({ length: columns }, () => {
      const line = new THREE.Line(lineGeometry(), new THREE.LineBasicMaterial({ transparent: true, opacity: 0.32 }));
      line.frustumCulled = false;
      this.root.add(line);
      return line;
    });
    this.reuseArcs = Array.from({ length: columns }, () => {
      const line = new THREE.Line(lineGeometry(),
        new THREE.LineDashedMaterial({ color: 0xfbd332, dashSize: 0.18, gapSize: 0.12, transparent: true, opacity: 0.6 }));
      line.visible = false;
      line.frustumCulled = false;
      this.root.add(line);
      return line;
    });
  }

  setSpeed(speed) {
    this.speed = speed;
  }

  // Move to position t. With animate, the rows fly out to the wall and back into e_t.
  show(t, animate) {
    if (!this.trace || !this.markers) return;
    this.position = t;
    const columns = this.trace.layers[this.layerIndex].columns;
    const lookups = this.index[this.layerIndex].lookups;
    const maxN = this.trace.model.max_ngram_size;
    this.followX = this.tileX(t);

    // the window around t, plus the earlier tokens the reuse arcs start from
    const earlier = new Set(lookups[t].map((x) => x.firstSeen));
    this.pads.forEach((tile) => (tile.visible = t < TOKEN_WINDOW));
    this.tiles.forEach((tile, i) => {
      const active = i === t;
      tile.visible = Math.abs(i - t) <= TOKEN_WINDOW || earlier.has(i);
      tile.material.emissive.set(active ? 0x6a5410 : 0x000000);
      tile.position.y = active ? 0.18 : 0;
      const element = tile.children[0].element;
      element.classList.toggle('active', active);
      element.classList.toggle('in-ngram', !active && i <= t && i > t - maxN);
    });
    this.brackets.forEach((bar, k) => {
      const span = ngramPositions(t, bar.userData.n);
      const left = this.tileX(span[0]) - TILE.width / 2;
      const right = this.tileX(t) + TILE.width / 2;
      bar.scale.x = right - left;
      bar.position.set((left + right) / 2, -0.5 - k * 0.24, 3.05);
    });

    this.markers.count = (t + 1) * columns.length;

    this.et.position.set(this.tileX(t), ET.y, 3);
    const gates = this.trace.gates?.[t]?.[this.layerIndex];
    this.gateBars.forEach((bar, k) => {
      bar.scale.y = Math.max(gates[k], 0.001) * 0.9;
      bar.position.y = (bar.scale.y - ET.height) / 2;
    });

    const tile = new THREE.Vector3(this.tileX(t), 0.5, 3);
    this.flight = lookups[t].map(({ row, firstSeen }, c) => {
      const color = new THREE.Color(ngramColor(columns[c].n));
      this.particles[c].material.color.copy(color);
      this.etCells[c].material.color.copy(color);
      const wallPoint = this.rowPoint(c, row);
      this.current[c].position.copy(wallPoint);
      const earlier = firstSeen < t ? new THREE.Vector3(this.tileX(firstSeen), 0.4, 3) : null;
      return { color, wallPoint, tile: tile.clone(), earlier };
    });
    this.linksDirty = true;

    if (animate) {
      this.animation = { start: performance.now() / 1000, duration: 1.7 / this.speed };
    } else {
      this.animation = null;
      this.settle();
    }
  }

  settle() {
    this.particles.forEach((p) => (p.visible = false));
    this.current.forEach((m) => {
      m.visible = true;
      m.material.opacity = 0.5;
    });
    this.etCells.forEach((cell) => (cell.material.opacity = 1));
    this.linksDirty = true;
  }

  updateLinks(visible) {
    const wallOffset = new THREE.Vector3(this.wall.position.x, 0, 0);
    const point = new THREE.Vector3();
    const curve = (line, from, to, lift) => {
      const control = new THREE.Vector3((from.x + to.x) / 2, Math.max(from.y, to.y) + lift, (from.z + to.z) / 2);
      const positions = line.geometry.attributes.position;
      for (let i = 0; i <= CURVE_SEGMENTS; i += 1) {
        bezier(from, control, to, i / CURVE_SEGMENTS, point);
        positions.setXYZ(i, point.x, point.y, point.z);
      }
      positions.needsUpdate = true;
    };
    this.flight.forEach((f, c) => {
      const target = f.wallPoint.clone().add(wallOffset);
      const link = this.links[c];
      link.visible = visible;
      link.material.color.copy(f.color);
      if (visible) curve(link, f.tile, target, 2.2);
      const arc = this.reuseArcs[c];
      arc.visible = visible && f.earlier != null;
      if (arc.visible) {
        curve(arc, f.earlier, target, 3.2);
        arc.computeLineDistances();
      }
    });
  }

  frame() {
    // keep the wall and the camera centred on the current token
    const dx = (this.followX - this.wall.position.x) * 0.12;
    const moving = Math.abs(dx) > 1e-4;
    if (moving) {
      this.wall.position.x += dx;
      this.camera.position.x += dx;
      this.controls.target.x += dx;
    }
    if (this.animation && this.flight) {
      const s = (performance.now() / 1000 - this.animation.start) / this.animation.duration;
      const out = smooth(clamp01((s - 0.05) / 0.45));
      const back = smooth(clamp01((s - 0.6) / 0.35));
      const wallOffset = new THREE.Vector3(this.wall.position.x, 0, 0);
      const cell = new THREE.Vector3();
      this.flight.forEach((f, c) => {
        const target = f.wallPoint.clone().add(wallOffset);
        const particle = this.particles[c];
        particle.visible = s < 0.97;
        if (s < 0.55) {
          const control = new THREE.Vector3((f.tile.x + target.x) / 2, target.y + 2.2, (f.tile.z + target.z) / 2);
          bezier(f.tile, control, target, out, particle.position);
        } else {
          this.etCells[c].getWorldPosition(cell);
          const control = new THREE.Vector3((target.x + cell.x) / 2, target.y + 1.2, (target.z + cell.z) / 2);
          bezier(target, control, cell, back, particle.position);
        }
        this.current[c].visible = s > 0.45;
        this.current[c].material.opacity = 0.5 + 0.5 * Math.max(0, 1 - Math.abs(s - 0.55) * 6);
        this.etCells[c].material.opacity = back > 0.98 ? 1 : 0.12;
      });
      if (this.linksDirty) this.updateLinks(false);
      this.linksDirty = false;
      if (s >= 1) {
        this.animation = null;
        this.settle();
        this.onStepEnd?.();
      }
    } else if (this.flight && (this.linksDirty || moving)) {
      this.updateLinks(true);
      this.linksDirty = false;
    }
    this.controls.update();
    this.renderer.render(this.scene, this.camera);
    this.labels.render(this.scene, this.camera);
  }
}
