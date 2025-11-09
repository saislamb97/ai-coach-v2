// components/AvatarCanvas.tsx
import * as React from 'react'
import * as THREE from 'three'
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js'
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js'

type Props = {
  glbUrl: string
  getVisemeFrame: () => number[] | null
  className?: string
  height?: number | string
  zoom?: number
}

/** ARKIT-15 canonical names (indices must match server) */
const ARKIT = [
  'jawOpen','mouthFunnel','mouthClose','mouthPucker',
  'mouthSmileLeft','mouthSmileRight','mouthLeft','mouthRight',
  'mouthFrownLeft','mouthFrownRight','mouthDimpleLeft','mouthDimpleRight',
  'mouthStretchLeft','mouthStretchRight','tongueOut'
] as const

/** tolerant alias map (case/underscore insensitive) */
const ALIASES: Record<string, string[]> = {
  jawopen: ['jawopen','mouthopen','viseme_aa','jaw_open'],
  mouthfunnel: ['mouthfunnel','lipsfunnel','viseme_ow','viseme_ou','viseme_oh','mouth_funnel'],
  mouthclose: ['mouthclose','lipsclose','mouth_close','viseme_m','viseme_b','viseme_p'],
  mouthpucker: ['mouthpucker','lipspucker','viseme_uw','viseme_w','mouth_pucker'],
  mouthsmileleft: ['mouthsmileleft','mouthsmile','mouth_smileleft','mouth_smile_left'],
  mouthsmileright: ['mouthsmileright','mouth_smile_right','mouth_smileright'],
  mouthleft: ['mouthleft','mouth_left'],
  mouthright: ['mouthright','mouth_right'],
  mouthfrownleft: ['mouthfrownleft','mouth_frownleft','mouth_frown_left'],
  mouthfrownright: ['mouthfrownright','mouth_frownright','mouth_frown_right'],
  mouthdimpleleft: ['mouthdimpleleft','mouth_dimpleleft','mouth_dimple_left'],
  mouthdimpleright: ['mouthdimpleright','mouth_dimpleright','mouth_dimple_right'],
  mouthstretchleft: ['mouthstretchleft','mouth_stretchleft','mouth_stretch_left','viseme_s','viseme_z'],
  mouthstretchright: ['mouthstretchright','mouth_stretchright','mouth_stretch_right','viseme_s','viseme_z'],
  tongueout: ['tongueout','tongue','tongue_out']
}
const lcKey = (s: string) => s.toLowerCase().replace(/[\s_-]+/g, '')

/* ---------------- precise head-first framing ---------------- */
const HEAD_NAMES = ['HeadTop_End','HeadTop','Head','J_Bip_C_Head','mixamorigHead','CC_Base_Head','HED','head'].map(s=>s.toLowerCase())
const NECK_NAMES = ['Neck','J_Bip_C_Neck','mixamorigNeck','CC_Base_Neck','neck'].map(s=>s.toLowerCase())

function findNode(root: THREE.Object3D, namesLower: string[]): THREE.Object3D | null {
  let hit: THREE.Object3D | null = null
  root.traverse(o => {
    if (hit) return
    const n = (o.name || '').toLowerCase()
    if (namesLower.some(c => n === c || n.endsWith(c))) hit = o
  })
  return hit
}

function frameFaceStrict(
  avatarRoot: THREE.Object3D,
  camera: THREE.PerspectiveCamera,
  controls: OrbitControls,
  zoom: number
) {
  avatarRoot.updateMatrixWorld(true)
  camera.updateMatrixWorld(true)

  const bboxAll = new THREE.Box3().setFromObject(avatarRoot)
  const H = Math.max(0.01, bboxAll.max.y - bboxAll.min.y)

  const head = findNode(avatarRoot, HEAD_NAMES)
  const neck = findNode(avatarRoot, NECK_NAMES)

  const headPos = new THREE.Vector3()
  const neckPos = new THREE.Vector3()
  const centerXZ = new THREE.Vector3((bboxAll.min.x + bboxAll.max.x)/2, 0, (bboxAll.min.z + bboxAll.max.z)/2)

  let target = new THREE.Vector3(centerXZ.x, bboxAll.max.y - H*0.06, centerXZ.z)
  let span   = Math.min(0.28, H * 0.24)

  if (head) head.getWorldPosition(headPos)
  if (neck) neck.getWorldPosition(neckPos)

  if (head) {
    if (neck) span = Math.max(0.12, Math.min(0.35, (headPos.y - neckPos.y) * 1.6))
    target.set(headPos.x, headPos.y - span*0.18, headPos.z)
  } else {
    const top = bboxAll.max.y
    const shoulder = bboxAll.min.y + H * 0.70
    span = Math.max(0.12, Math.min(0.35, (top - shoulder) * 1.4))
    target.set(centerXZ.x, top - span*0.55, centerXZ.z)
  }

  span = span / Math.max(zoom, 0.001)
  const dist = (span * 0.5) / Math.tan(THREE.MathUtils.degToRad(camera.fov * 0.5))

  controls.target.copy(target)
  camera.position.set(target.x, target.y, target.z + Math.max(0.28, dist))
  camera.updateProjectionMatrix()
  controls.minDistance = Math.max(0.20, dist * 0.4)
  controls.maxDistance = Math.max(0.8,  dist * 2.5)
  controls.update()
}

/* ---------------- component ---------------- */
export default function AvatarCanvas({
  glbUrl,
  getVisemeFrame,
  className = '',
  height = 180,
  zoom = 2
}: Props) {
  const hostRef = React.useRef<HTMLDivElement>(null)

  const three = React.useRef<{
    scene: THREE.Scene
    camera: THREE.PerspectiveCamera
    renderer: THREE.WebGLRenderer
    controls: OrbitControls
  } | null>(null)
  const arkitTargets = React.useRef<Array<{
    influences: number[]
    controlled: number[]
    mapIdx: Int16Array
  }>>([])

  // Minimal smoothing just to prevent tiny float noise — keep tiny.
  const vis = React.useRef<Float32Array>(new Float32Array(15))
  const ema = 0.06   // small; scheduler already smooths

  React.useEffect(() => {
    const el = hostRef.current!
    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0xf8fafc)

    const w = el.clientWidth, h = Math.max(1, el.clientHeight)
    const camera = new THREE.PerspectiveCamera(26, w/h, 0.1, 100)
    camera.position.set(0, 1.55, 1.0)

    const renderer = new THREE.WebGLRenderer({ antialias: true })
    renderer.outputColorSpace = THREE.SRGBColorSpace
    renderer.toneMapping = THREE.ACESFilmicToneMapping
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
    renderer.setSize(w, h)
    el.innerHTML = ''
    el.appendChild(renderer.domElement)

    // Light rig
    scene.add(new THREE.HemisphereLight(0xffffff, 0x8899aa, 0.35))
    const key  = new THREE.DirectionalLight(0xffffff, 1.0); key.position.set( 1, 1.8,  2.2); scene.add(key)
    const fill = new THREE.DirectionalLight(0xffffff, 0.7); fill.position.set(-1.6, 1.4,  0.6); scene.add(fill)
    const rim  = new THREE.DirectionalLight(0xffffff, 0.45); rim.position.set( 0.6, 1.7, -2.2); scene.add(rim)

    const controls = new OrbitControls(camera, renderer.domElement)
    controls.enablePan = false
    controls.enableDamping = true
    controls.minDistance = 0.32
    controls.maxDistance = 2.5
    controls.target.set(0, 1.55, 0)

    three.current = { scene, camera, renderer, controls }

    const ro = new ResizeObserver(() => {
      const w = el.clientWidth, h = Math.max(1, el.clientHeight)
      camera.aspect = w / h
      camera.updateProjectionMatrix()
      renderer.setSize(w, h)
    })
    ro.observe(el)

    // load avatar
    const loader = new GLTFLoader().setCrossOrigin('anonymous')
    loader.load(glbUrl, (gltf) => {
      const root = gltf.scene || gltf.scenes?.[0]
      if (!root) return
      scene.add(root)
      scene.updateMatrixWorld(true)
      root.updateMatrixWorld(true)

      // map ARKit indices
      arkitTargets.current = []
      let meshesMapped = 0
      root.traverse((obj: any) => {
        if (!(obj?.isMesh && obj.morphTargetDictionary && obj.morphTargetInfluences)) return
        const dict: Record<string, number> = obj.morphTargetDictionary
        const dictLc: Record<string, number> = {}
        for (const [k, v] of Object.entries(dict)) dictLc[lcKey(k)] = v

        const mapIdx = new Int16Array(15).fill(-1)
        const controlled: number[] = []
        for (let i = 0; i < ARKIT.length; i++) {
          const canon = lcKey(ARKIT[i])
          let idx = dictLc[canon]
          if (idx == null) {
            const alts = (ALIASES as any)[canon] || []
            for (const alt of alts) { idx = dictLc[lcKey(alt)]; if (idx != null) break }
          }
          if (idx != null) { mapIdx[i] = idx; controlled.push(idx) }
        }
        if (controlled.length) {
          arkitTargets.current.push({ influences: obj.morphTargetInfluences as number[], controlled, mapIdx })
          meshesMapped++
        }
      })

      if (!arkitTargets.current.length) {
        console.warn('[AvatarCanvas] ⚠️ No ARKit-compatible morph targets found. Check RPM export/names.')
      } else {
        // eslint-disable-next-line no-console
        console.log(`[AvatarCanvas] ARKit morphs mapped on ${meshesMapped} mesh(es).`)
      }

      try {
        frameFaceStrict(root, camera, controls, zoom)
        requestAnimationFrame(() => frameFaceStrict(root, camera, controls, zoom))
      } catch (e) {
        console.warn('[AvatarCanvas] framing error', e)
      }
    }, undefined, (err) => {
      console.error('[AvatarCanvas] GLB load error:',
        err instanceof Error ? err.message : String(err)
      )
    })

    // render loop
    let raf = 0
    const loop = () => {
      if (typeof renderer.setAnimationLoop === 'function') {
        renderer.setAnimationLoop(loop)
      } else {
        raf = requestAnimationFrame(loop)
      }
      controls.update()

      const f = getVisemeFrame()
      if (Array.isArray(f) && f.length >= 15) {
        // Small EMA only; otherwise pass-through (server owns shaping)
        for (let i = 0; i < 15; i++) {
          const t = Math.max(0, Math.min(1, f[i] || 0))
          vis.current[i] = vis.current[i] * ema + t * (1 - ema)
        }
      } else {
        // mild relaxation (if stream paused)
        for (let i = 0; i < 15; i++) vis.current[i] *= 0.94
      }

      // write influences
      if (arkitTargets.current.length) {
        for (const t of arkitTargets.current) {
          for (let k = 0; k < t.controlled.length; k++) t.influences[t.controlled[k]] = 0
          for (let i = 0; i < 15; i++) {
            const idx = t.mapIdx[i]
            if (idx >= 0) t.influences[idx] = vis.current[i]
          }
        }
      }

      renderer.render(scene, camera)
    }
    if (three.current?.renderer.setAnimationLoop) {
      three.current.renderer.setAnimationLoop(loop as any)
    } else {
      raf = requestAnimationFrame(loop)
    }

    return () => {
      if (three.current?.renderer.setAnimationLoop) three.current.renderer.setAnimationLoop(null as any)
      cancelAnimationFrame(raf)
      ro.disconnect()
      try {
        // best-effort disposal
        three.current?.scene.traverse((obj: any) => {
          if (obj.isMesh) {
            obj.geometry?.dispose?.()
            if (obj.material) {
              const mats = Array.isArray(obj.material) ? obj.material : [obj.material]
              for (const m of mats) {
                m.map?.dispose?.(); m.lightMap?.dispose?.(); m.aoMap?.dispose?.()
                m.emissiveMap?.dispose?.(); m.bumpMap?.dispose?.(); m.normalMap?.dispose?.()
                m.roughnessMap?.dispose?.(); m.metalnessMap?.dispose?.(); m.envMap?.dispose?.()
                m.dispose?.()
              }
            }
          }
        })
        three.current?.renderer.dispose()
      } catch {}
      three.current = null
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [glbUrl, getVisemeFrame, zoom])

  const h = typeof height === 'number' ? `${height}px` : (height || '320px')
  return (
    <div
      ref={hostRef}
      style={{ height: h }}
      className={`w-full rounded-lg border border-slate-200 bg-slate-100 overflow-hidden ${className || ''}`}
    />
  )
}
