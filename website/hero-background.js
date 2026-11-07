/* A procedural illustration of scientific computer use. No live execution data. */
(() => {
  'use strict';
  const canvas = document.getElementById('science-background');
  const hero = document.querySelector('.hero');
  if (!canvas || !hero) return;
  const ctx = canvas.getContext('2d');
  if (!ctx) return;
  const mask = document.createElement('canvas');
  const mc = mask.getContext('2d');
  const reduced = matchMedia('(prefers-reduced-motion: reduce)');
  const mono = '"Cascadia Code",Consolas,monospace';
  const TAU = Math.PI * 2;
  const blue = '37,91,206', cyan = '0,164,192', violet = '107,76,214';
  const pointer = { x: 0, y: 0, active: false, strength: 0 };
  let width = 0, height = 0, dpr = 1, time = reduced.matches ? 5700 : 0;
  let last = 0, raf = null, paused = reduced.matches, visible = true;
  let section = 0;
  let scene = {}, objects = {}, textTop = 0;
  const clamp = (n, a = 0, b = 1) => Math.min(b, Math.max(a, n));
  const smooth = n => { n = clamp(n); return n * n * (3 - 2 * n); };
  const lerp = (a, b, t) => a + (b - a) * t;
  const pointMix = (a, b, t) => ({ x: lerp(a.x, b.x, t), y: lerp(a.y, b.y, t) });

  function line(points, color = blue, alpha = .4, size = 1, dash = []) {
    ctx.beginPath(); points.forEach((p, i) => i ? ctx.lineTo(p.x, p.y) : ctx.moveTo(p.x, p.y));
    ctx.strokeStyle = `rgba(${color},${alpha})`; ctx.lineWidth = size;
    ctx.setLineDash(dash); ctx.stroke(); ctx.setLineDash([]);
  }
  function polygon(points, color, alpha) {
    ctx.beginPath(); points.forEach((p, i) => i ? ctx.lineTo(p.x, p.y) : ctx.moveTo(p.x, p.y));
    ctx.closePath(); ctx.fillStyle = `rgba(${color},${alpha})`; ctx.fill();
  }
  function label(value, x, y, size = 9, alpha = .7, color = blue) {
    ctx.font = `${size}px ${mono}`; ctx.fillStyle = `rgba(${color},${alpha})`; ctx.fillText(value, x, y);
  }
  function dot(p, size = 2, color = cyan, alpha = 1) {
    ctx.beginPath(); ctx.arc(p.x, p.y, size, 0, TAU); ctx.fillStyle = `rgba(${color},${alpha})`; ctx.fill();
  }
  function bloom(p, radius, color = cyan, alpha = .18) {
    const g = ctx.createRadialGradient(p.x, p.y, 0, p.x, p.y, radius);
    g.addColorStop(0, `rgba(${color},${alpha})`); g.addColorStop(1, `rgba(${color},0)`);
    ctx.fillStyle = g; ctx.fillRect(p.x - radius, p.y - radius, radius * 2, radius * 2);
  }
  function corners(x, y, w, h, color = cyan, alpha = .9, size = 12) {
    for (const sx of [-1, 1]) for (const sy of [-1, 1]) {
      const p = { x: x + (sx > 0 ? w : 0), y: y + (sy > 0 ? h : 0) };
      line([{ x: p.x - sx * size, y: p.y }, p, { x: p.x, y: p.y - sy * size }], color, alpha, 1.4);
    }
  }
  function cross(p, size = 5, alpha = .5) {
    line([{ x: p.x - size, y: p.y }, { x: p.x + size, y: p.y }], blue, alpha, .8);
    line([{ x: p.x, y: p.y - size }, { x: p.x, y: p.y + size }], blue, alpha, .8);
  }
  function curve(points, color, alpha, thickness = 1, dash = []) {
    ctx.beginPath(); ctx.moveTo(points[0].x, points[0].y);
    ctx.bezierCurveTo(points[1].x, points[1].y, points[2].x, points[2].y, points[3].x, points[3].y);
    ctx.strokeStyle = `rgba(${color},${alpha})`; ctx.lineWidth = thickness;
    ctx.setLineDash(dash); ctx.stroke(); ctx.setLineDash([]);
  }
  function onCurve(points, t) {
    const v = 1 - t;
    return { x: v ** 3 * points[0].x + 3 * v * v * t * points[1].x + 3 * v * t * t * points[2].x + t ** 3 * points[3].x,
      y: v ** 3 * points[0].y + 3 * v * v * t * points[1].y + 3 * v * t * t * points[2].y + t ** 3 * points[3].y };
  }

  function fit() {
    width = document.documentElement.clientWidth;
    height = hero.getBoundingClientRect().height;
    dpr = Math.min(devicePixelRatio || 1, 1.75);
    canvas.style.setProperty('--science-background-width', `${width}px`);
    canvas.width = Math.round(width * dpr); canvas.height = Math.round(height * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    mask.width = canvas.width; mask.height = canvas.height;
    mc.setTransform(dpr, 0, 0, dpr, 0, 0);
    const origin = canvas.getBoundingClientRect();
    const copy = hero.querySelector('.hero-copy').getBoundingClientRect();
    textTop = copy.top - origin.top;
    const protectedNodes = [];
    for (const selector of ['h1', '.hero-text', '.author-list', '.affiliation-list', '.author-note', '.logo-row img', '.hero-actions .button']) {
      for (const el of hero.querySelectorAll(selector)) protectedNodes.push({el,selector});
    }
    for (const {el,selector} of protectedNodes) {
      let r = el.getBoundingClientRect();
      if (selector === 'h1' || selector === '.author-note') { const range = document.createRange(); range.selectNodeContents(el); r = range.getBoundingClientRect(); }
      // Feather each original text block; the page typography and layout never move.
      mc.save();
      mc.translate(r.x - origin.x + r.width / 2, r.y - origin.y + r.height / 2);
      mc.scale(r.width / 2 + (selector === 'h1' ? 90 : 100), r.height / 2 + 65);
      const g = mc.createRadialGradient(0, 0, 0, 0, 0, 1);
      g.addColorStop(0, 'rgba(0,0,0,1)'); g.addColorStop(.67, 'rgba(0,0,0,.99)');
      g.addColorStop(.82, 'rgba(0,0,0,.9)'); g.addColorStop(1, 'rgba(0,0,0,0)');
      mc.fillStyle = g; mc.fillRect(-1, -1, 2, 2); mc.restore();
    }
    const mobile = width < 700;
    // Give every scientific object room around the unchanged academic content.
    // Reduce the surface a further 20% for the mentor review composition.
    scene = { x: width * (mobile ? .91 : .85), y: height * (mobile ? .2 : .26),
      scale: mobile ? width * .1624 : Math.min(width * .06832, height * .14) };
    const unit = Math.min(width / 1440, 1.2);
    const topBand = Math.max(58, Math.min(125, textTop * .46));
    objects = {
      ligand: { x: width * .15, y: height * .235, scale: 40 * unit },
      purine: { x: width * .255, y: height * .13, scale: 30 * unit },
      orbit: { x: width * .902, y: height * .68, scale: 94 * unit },
      map: { x: width * .105, y: height * .72, scale: 108 * unit },
      proof: { x: width * .365, y: topBand - 12, scale: 1 * unit },
      typeset: { x: width * .638, y: topBand - 12, scale: 1 * unit }
    };
    render();
  }

  function phases() {
    const t = (time % 12000) / 1000;
    section = Math.min(3, Math.floor(t / 3));
    return { t, step: section, p: t % 3 / 3,
      rotation: smooth((t - 6) / 2.5) * .68 - smooth((t - 11) / 1) * .68,
      selected: smooth((t - 3) / .7) * (1 - smooth((t - 11) / 1)),
      feedback: smooth((t - 9) / .4) * (1 - smooth((t - 11.5) / .5)) };
  }
  function project(u, v, z = 0, phase = phases()) {
    const yaw = -.46 + phase.rotation;
    const x = u * Math.cos(yaw) - v * Math.sin(yaw);
    const y = u * Math.sin(yaw) + v * Math.cos(yaw);
    return { x: scene.x + x * scene.scale, y: scene.y + (y * .55 - z * .96) * scene.scale };
  }
  function surfaceZ(u, v) {
    // Uses the homepage's documented scientific plotting example: sin(x) * sin(y).
    return Math.sin(u * 2.6) * Math.sin(v * 2.6) * .49;
  }

  function drawingField() {
    const haze = ctx.createLinearGradient(0, 0, width, height);
    haze.addColorStop(0, '#f6f9ff'); haze.addColorStop(.35, '#ffffff'); haze.addColorStop(1, '#eaf5ff');
    ctx.fillStyle = haze; ctx.fillRect(0, 0, width, height);
    const pitch = width < 700 ? 36 : 48;
    for (let x = 24; x < width; x += pitch) for (let y = 24; y < height; y += pitch) {
      const edge = Math.abs(x / width - .5) * 2;
      dot({ x, y }, .7, blue, .045 + edge * .12);
    }
    if (width < 700) return;
    // Subpixel registration ticks, rather than decorative star particles.
    for (let y = 46; y < height - 35; y += 17) {
      const long = Math.round((y - 46) / 17) % 5 === 0;
      line([{ x: width - 23, y }, { x: width - (long ? 37 : 29), y }], blue, long ? .35 : .18, .7);
    }
  }

  function coordinatePlane(phase) {
    const boundary = [[-1.22,-1.22],[1.22,-1.22],[1.22,1.22],[-1.22,1.22]].map(p => project(...p, -.58, phase));
    polygon(boundary, blue, .045);
    line([...boundary, boundary[0]], blue, .26, .8);
    for (let i = 0; i <= 20; i++) {
      const u = -1.2 + i * .12;
      line([project(u, -1.2, -.58, phase), project(u, 1.2, -.58, phase)], blue, i % 5 === 0 ? .23 : .1, .6);
      line([project(-1.2, u, -.58, phase), project(1.2, u, -.58, phase)], blue, i % 5 === 0 ? .23 : .1, .6);
    }
    const origin = project(-1.22, 1.22, -.58, phase);
    for (const end of [project(-.57, 1.22, -.58, phase), project(-1.22, .57, -.58, phase), project(-1.22, 1.22, .12, phase)]) {
      line([origin, end], blue, .6, 1); cross(end, 3, .6);
    }
    if (width > 700) {
      const p = project(-.6, 1.26, -.58, phase);
      label('x', p.x + 3, p.y + 12, 10, .7);
      const q = project(-1.3, .55, -.58, phase);
      label('y', q.x - 7, q.y, 10, .7);
    }
  }

  function scientificSurface(phase) {
    const resolution = width < 700 ? 22 : 29;
    const rows = [], quads = [];
    for (let j = 0; j <= resolution; j++) {
      const row = [];
      for (let i = 0; i <= resolution; i++) {
        const u = i / resolution * 2 - 1, v = j / resolution * 2 - 1;
        const z = surfaceZ(u, v), p = project(u, v, z, phase);
        const item = { ...p, z, u, v, i, j };
        row.push(item);
      }
      rows.push(row);
    }
    for (let j = 0; j < resolution; j++) for (let i = 0; i < resolution; i++) {
      const points = [rows[j][i], rows[j][i+1], rows[j+1][i+1], rows[j+1][i]];
      quads.push({ points, depth: (i - j * .45), z: points.reduce((sum,p) => sum + p.z, 0) / 4 });
    }
    quads.sort((a,b) => a.depth - b.depth);
    for (const q of quads) {
      const light = (q.z + .5);
      const rgb = `${Math.round(18 + light * 15)},${Math.round(65 + light * 98)},${Math.round(151 + light * 64)}`;
      polygon(q.points, rgb, .74);
      line([...q.points, q.points[0]], '130,228,255', .32 + light * .25, .55);
    }
    // Sharp contour edges and ruled projections make this a manipulable object in a coordinate space.
    for (const row of [rows[0], rows[resolution]]) line(row, blue, .8, 1.15);
    line(rows.map(row => row[0]), cyan, .9, 1.3);
    line(rows.map(row => row[resolution]), cyan, .9, 1.3);
    for (let i = 0; i <= resolution; i += 3) {
      const p = rows[resolution][i], base = project(p.u, p.v, -.58, phase);
      line([p, base], blue, .15, .65);
    }
    // A scan plane samples actual points of the same surface, then exposes an image-space selection.
    const scanU = -1 + ((time / 4100) % 1) * 2;
    const scanPoints = [];
    for (let k = 0; k <= 50; k++) {
      const v = k / 25 - 1; scanPoints.push(project(scanU, v, surfaceZ(scanU, v), phase));
    }
    polygon([project(scanU,-1.1,-.58,phase),project(scanU,1.1,-.58,phase),project(scanU,1.1,.67,phase),project(scanU,-1.1,.67,phase)], cyan, .065);
    line(scanPoints, '0,230,255', .95, 2.2);
    for (let i = 0; i < scanPoints.length; i += 8) { dot(scanPoints[i], 1.5, '185,255,255', .95); }
    const middle = scanPoints[25]; bloom(middle, scene.scale * .55, cyan, .1);
    const anchor = project(.57, .57, surfaceZ(.57,.57), phase);
    const detection = project(-.57,-.57,surfaceZ(-.57,-.57),phase);
    const size = scene.scale * .31;
    corners(anchor.x-size/2,anchor.y-size/2,size,size,cyan,.58 + phase.selected * .4,9);
    corners(detection.x-size*.4,detection.y-size*.4,size*.8,size*.8,blue,.45,7);
    cross(anchor, 5, .85);
    if (width > 700) {
      label('z = sin(x) · sin(y)', scene.x - scene.scale * .9, scene.y - scene.scale * 1.02 - 9, 10, .7);
    }
    return anchor;
  }

  function perceptionRaster(phase) {
    if (width < 700) return;
    const x0 = scene.x - scene.scale * 1.15, y0 = scene.y - scene.scale * .9, cell = Math.min(4.5, width / 280);
    // A single image-space sampling strip resolves into geometric information.
    for (let j = 0; j < 3; j++) for (let i = 0; i < 18; i++) {
      const wave = Math.sin(i * .37 + j * .68), keep = Math.sin(i * 13.7 + j * 9.1);
      if (keep > .5 || (i < 4 && j > 2)) continue;
      let alpha = (.14 + (wave + 1) * .15) * (i / 27 * .65 + .35);
      const scan = (time * .003) % 27; if (Math.abs(i - scan) < 1.2) alpha += .35;
      ctx.fillStyle = `rgba(${i > 18 ? cyan : blue},${alpha})`;
      ctx.fillRect(x0 + i * cell, y0 + j * cell, cell - 1.5, cell - 1.5);
    }
    const start = { x: x0 + 18 * cell, y: y0 + 3 * cell };
    const end = project(.5,-1.05,.1,phase);
    line([start,{x:start.x+18,y:start.y},end],blue,.22,.8,[3,5]);
  }

  function objectPoint(rig, p, angle = 0, tilt = -.18) {
    const a = angle;
    const x = p.x*Math.cos(a)+p.z*Math.sin(a), z=-p.x*Math.sin(a)+p.z*Math.cos(a);
    const y = p.y*Math.cos(tilt)-z*Math.sin(tilt), depth=p.y*Math.sin(tilt)+z*Math.cos(tilt);
    const roll=rig.roll||0, k=3.8/(3.8+depth*.14);
    return {x:rig.x+(x*Math.cos(roll)-y*Math.sin(roll))*rig.scale*k,
      y:rig.y+(x*Math.sin(roll)+y*Math.cos(roll))*rig.scale*k,depth};
  }
  function sphere(p, radius, color, alpha = 1) {
    const g=ctx.createRadialGradient(p.x-radius*.34,p.y-radius*.38,radius*.05,p.x,p.y,radius);
    g.addColorStop(0,`rgba(237,252,255,${alpha})`);
    g.addColorStop(.3,`rgba(${color},${alpha})`);g.addColorStop(1,`rgba(${color},${alpha*.64})`);
    ctx.fillStyle=g;ctx.beginPath();ctx.arc(p.x,p.y,radius,0,TAU);ctx.fill();
  }
  function inspection(p, size, color, alpha = .65) {
    corners(p.x-size/2,p.y-size/2,size,size,color,alpha,Math.min(8,size*.23));
  }

  function drawMolecule(atoms,bonds,doubleBonds,rig,angle) {
    const points=atoms.map(a=>({...objectPoint(rig,a,angle),e:a.e}));
    ctx.lineCap='round';
    for(const [a,b] of bonds){line([points[a],points[b]],'91,128,177',.6,3);line([points[a],points[b]],'181,221,246',.5,1);}
    for(const [a,b] of doubleBonds){
      const A=points[a],B=points[b],len=Math.hypot(B.x-A.x,B.y-A.y),nx=-(B.y-A.y)/len*3.5,ny=(B.x-A.x)/len*3.5;
      line([{x:A.x+nx,y:A.y+ny},{x:B.x+nx,y:B.y+ny}],'91,128,177',.5,1.3);
    }
    ctx.lineCap='butt';
    const atomColors={C:'65,101,156',N:'65,120,229',O:'226,105,130'};
    for(const p of [...points].sort((a,b)=>b.depth-a.depth)){sphere(p, p.e==='C'?4.5:5.5,atomColors[p.e],.9);}
    return points;
  }

  function molecularModels() {
    if(width<700)return;
    // Keep the selected amino-acid-like model and add one smaller purine schematic.
    const atoms=[];
    for(let i=0;i<6;i++)atoms.push({x:Math.cos(i*TAU/6)*.56+.52,y:Math.sin(i*TAU/6)*.56,z:0,e:'C'});
    atoms.push({x:1.62,y:0,z:0,e:'O'},{x:-.57,y:0,z:.12,e:'C'},
      {x:-1.05,y:.4,z:-.1,e:'C'},{x:-1.04,y:1,z:.05,e:'N'},
      {x:-1.64,y:.06,z:.06,e:'C'},{x:-2.17,y:.35,z:.03,e:'O'},{x:-1.67,y:-.55,z:.06,e:'O'});
    const bonds=[[0,1],[1,2],[2,3],[3,4],[4,5],[5,0],[0,6],[3,7],[7,8],[8,9],[8,10],[10,11],[10,12]];
    const points=drawMolecule(atoms,bonds,[[0,1],[2,3],[4,5],[10,11]],{...objects.ligand,roll:-.3},-.45+Math.sin(time*.00015)*.5);
    inspection(points[9],24,violet,.4);
    const purine=[[-.9,.5,'C'],[-1.2,-.05,'N'],[-.85,-.65,'C'],[-.2,-.65,'N'],[.15,-.05,'C'],[-.25,.5,'C'],
      [.83,-.05,'N'],[.95,.58,'C'],[.37,.95,'N'],[-1.12,-1.22,'N']].map(([x,y,e])=>({x,y,z:0,e}));
    const purineBonds=[[0,1],[1,2],[2,3],[3,4],[4,5],[5,0],[4,6],[6,7],[7,8],[8,5],[2,9]];
    drawMolecule(purine,purineBonds,[[0,1],[2,3],[4,5],[6,7]],{...objects.purine,roll:.38},.35+Math.sin(time*.00012+1)*.4);
  }

  function celestialOrbits() {
    if(width<700)return;
    const r=objects.orbit,s=r.scale,tilt=-.35,angle=time*.00036;
    const orbitPoint=a=>({x:r.x+Math.cos(a)*s,y:r.y+Math.sin(a)*s*.39+Math.cos(a)*s*tilt});
    const orbit=Array.from({length:97},(_,i)=>orbitPoint(i/96*TAU));
    line(orbit,blue,.34,.9);
    const farOrbit=orbit.map(p=>({x:r.x+(p.x-r.x)*1.24,y:r.y+(p.y-r.y)*1.24}));
    line(farOrbit,blue,.15,.7,[2,5]);
    const sun={x:r.x-s*1.4,y:r.y-s*.13};
    bloom(sun,26,'244,184,86',.14);sphere(sun,7,'235,174,86',.9);
    polygon([{x:r.x,y:r.y-s*.24},{x:r.x+s*1.4,y:r.y-s*.08},{x:r.x+s*1.4,y:r.y+s*.22},{x:r.x,y:r.y+s*.24}],blue,.045);
    for(let i=-1;i<=1;i++)line([{x:sun.x+14,y:sun.y+i*12},{x:r.x-s*.4,y:r.y+i*12}],cyan,.16,.7,[3,5]);
    const radius=s*.34;
    sphere(r,radius,'36,111,186',.9);
    ctx.save();ctx.beginPath();ctx.arc(r.x,r.y,radius,0,TAU);ctx.clip();
    for(let i=-2;i<=2;i++){
      ctx.beginPath();ctx.ellipse(r.x,r.y+i*radius*.27,radius*Math.sqrt(1-(i*.27)**2),radius*.17,0,0,TAU);
      ctx.strokeStyle='rgba(153,231,249,.42)';ctx.lineWidth=.7;ctx.stroke();
    }
    for(let i=0;i<5;i++){
      ctx.beginPath();ctx.ellipse(r.x,r.y,Math.abs(Math.sin(i*.65+time*.00008))*radius,radius,.15,0,TAU);
      ctx.strokeStyle='rgba(153,231,249,.3)';ctx.stroke();
    }
    ctx.restore();
    const moon=orbitPoint(angle);
    sphere(moon,6.5,'126,147,180',.95);inspection(moon,25,cyan,.7);
    const trail=[];for(let i=0;i<16;i++)trail.push(orbitPoint(angle-i*.025));line(trail,cyan,.7,1.8);
  }

  function geospatialMap() {
    if(width<700)return;
    const r=objects.map,s=r.scale;
    const mapPoint=(u,v,z=0)=>({x:r.x+(u+v*.24)*s,y:r.y+(v*.5-z*.65)*s});
    const outline=[[-1,-.84],[1,-.84],[1,.84],[-1,.84]].map(p=>mapPoint(...p));
    polygon(outline,'33,160,160',.03);
    for(let i=-4;i<=4;i++){
      line([mapPoint(i/4,-.84),mapPoint(i/4,.84)],cyan,.09,.65);
      line([mapPoint(-1,i/5),mapPoint(1,i/5)],cyan,.09,.65);
    }
    for(let level=0;level<13;level++){
      const contour=[],rad=.94-level*.056;
      for(let i=0;i<=90;i++){
        const a=i/90*TAU,k=1+.09*Math.sin(a*3+.3)+.07*Math.cos(a*5);
        contour.push(mapPoint(Math.cos(a)*rad*k,Math.sin(a)*rad*.78*k,level*.034));
      }
      if(level%3===0)polygon(contour,cyan,.027);
      line(contour,level%3===0?blue:cyan,.26+level*.018,level%3===0?1.05:.7);
    }
    const valley=[];for(let i=0;i<40;i++){const v=i/39*1.65-.85;valley.push(mapPoint(-.43+Math.sin(v*3)*.14,v,.03));}
    line(valley,blue,.55,1.3);
    const selected=[[.04,-.11],[.29,-.22],[.61,-.04],[.53,.31],[.22,.39],[-.02,.12]].map(p=>mapPoint(...p,.13));
    polygon(selected,cyan,.11);line([...selected,selected[0]],cyan,.85,1.25);
    selected.forEach(p=>dot(p,2,cyan,.8));
    const scan=((time*.00016)%1)*2-1;
    line([mapPoint(scan,-.9,.05),mapPoint(scan,.9,.05)],cyan,.65,1.3);
    const target=mapPoint(.3,.12,.13);inspection(target,25,cyan,.65);
  }

  function proofEvolution() {
    if(width<900)return;
    const r=objects.proof,s=r.scale,p=(time%12000)/12000;
    // Lagrange's identity: the difference of the two sides is (ad-bc)^2 >= 0.
    ctx.textAlign='center';
    label('(ac + bd)² ≤ (a² + b²)(c² + d²)',r.x,r.y,10.5*s,.8,blue);
    const root={x:r.x,y:r.y+13*s};
    const left={x:r.x-75*s,y:r.y+37*s},right={x:r.x+75*s,y:r.y+37*s};
    line([root,{x:r.x,y:r.y+21*s},left],violet,.25,.85);
    line([{x:r.x,y:r.y+21*s},right],violet,.25,.85);
    label('Δ = (ad−bc)²',left.x,left.y+14*s,9.5*s,p>.18?.76:.28);
    label('(ad−bc)² ≥ 0',right.x,right.y+14*s,9.5*s,p>.43?.76:.28);
    label('ring',left.x,left.y+31*s,8.5*s,p>.25?.66:.2,violet);
    label('sq_nonneg',right.x,right.y+31*s,8.5*s,p>.52?.66:.2,violet);
    if(p<.68)dot(pointMix(root,p<.34?left:right,(p% .34)/.34),2,violet,.7);
    if(p>.68){
      const end={x:r.x,y:r.y+88*s};
      line([{x:left.x,y:left.y+39*s},end,{x:right.x,y:right.y+39*s}],cyan,.28,.8);
      label('nlinarith',end.x,end.y+15*s,8.5*s,.7,cyan);
      line([{x:end.x-4,y:end.y},{x:end.x,y:end.y+4},{x:end.x+7,y:end.y-5}],cyan,.8,1.2);
    }
    ctx.textAlign='left';
  }

  function typesetting() {
    if(width<900)return;
    const r=objects.typeset,s=r.scale,p=(time%14000)/14000;
    const w=122*s,h=156*s,left=r.x-w/2,top=Math.max(38,textTop-h-50);
    const at=(x,y)=>({x:left+x*s,y:top+y*s});
    const paper=[at(0,0),at(111,0),at(122,11),at(122,156),at(0,156)];
    polygon(paper.map(q=>({x:q.x+3,y:q.y+4})),blue,.035);
    polygon(paper,'255,255,255',.87);line([...paper,paper[0]],blue,.24,.85);
    polygon([at(111,0),at(111,11),at(122,11)],blue,.06);
    line([at(111,0),at(111,11),at(122,11)],blue,.22,.6);
    label('RESEARCH NOTES',left+12*s,top+14*s,5.8*s,.58);
    ctx.font=`${8.5*s}px Georgia,serif`;ctx.fillStyle=`rgba(${blue},.8)`;
    ctx.fillText('Scientific workflows',left+12*s,top+28*s);
    line([at(12,35),at(110,35)],blue,.23,.7);
    label('ABSTRACT',left+12*s,top+46*s,5.5*s,.55);
    for(let i=0;i<2;i++)line([at(12,52+i*5),at(i?94:110,52+i*5)],blue,.19,.65);
    label('1  METHOD',left+12*s,top+71*s,5.8*s,.65);
    label('2  RESULTS',left+68*s,top+71*s,5.8*s,.65);
    // A paragraph is edited in place; the caret and subsequent lines follow the insertion.
    const words=['We study','reusable skills','for scientific','computer use.'];
    const progress=clamp((p-.1)/.62)*words.join('').length;
    let consumed=0,activeLine=0,activeCount=0;
    for(let i=0;i<words.length;i++){
      const count=clamp(Math.floor(progress-consumed),0,words[i].length);
      if(progress>=consumed){activeLine=i;activeCount=count;}
      if(count>0)label(words[i].slice(0,count),left+12*s,top+(83+i*7)*s,5.3*s,.74);
      consumed+=words[i].length;
    }
    if(p<.78){
      ctx.fillStyle=`rgba(${cyan},.055)`;
      ctx.fillRect(left+10*s,top+(77+activeLine*7)*s,46*s,8*s);
      if(Math.floor(time/550)%2===0){
        const x=12+activeCount*3.18,y=83+activeLine*7;
        line([at(x,y-5.5),at(x,y+1)],cyan,.82,.9);
      }
    }
    const reflow=smooth((p-.5)*3)*6;
    for(let i=0;i<4;i++)line([at(12,113+i*5+reflow),at(i===3?43:55,113+i*5+reflow)],blue,.2,.6);
    for(let i=0;i<3;i++)line([at(68,82+i*5),at(i===2?99:110,82+i*5)],blue,.18,.65);
    line([at(70,100),at(70,121),at(110,121)],blue,.32,.6);
    const plot=[];for(let i=0;i<=24;i++)plot.push(at(71+i*1.55,118-13*Math.exp(-(((i-15)/7)**2))));
    line(plot,cyan,.65,.9);
    for(let i=0;i<3;i++)line([at(68,130+i*5),at(i===2?98:110,130+i*5)],blue,.16,.6);
    label('1',left+w/2,top+148*s,5*s,.35);
  }

  function microCursor(p, alpha = 1) {
    ctx.save(); ctx.translate(p.x,p.y);
    polygon([{x:0,y:0},{x:0,y:17},{x:4.4,y:13},{x:8,y:21},{x:11.5,y:19},{x:7.4,y:11.5},{x:13,y:11}],blue,alpha);
    ctx.strokeStyle=`rgba(255,255,255,${alpha})`;ctx.lineWidth=1;ctx.stroke();ctx.restore();
  }
  function operation(anchor, phase) {
    const start={x:anchor.x-scene.scale*.55,y:anchor.y+scene.scale*.38};
    const drag={x:anchor.x+scene.scale*.25,y:anchor.y-scene.scale*.22};
    const path=[start,{x:anchor.x-scene.scale*.28,y:start.y+4},{x:anchor.x-scene.scale*.2,y:anchor.y+10},anchor];
    curve(path,blue,.55,1.25,[3,4]);
    line([anchor,drag],cyan,.5,1,[3,4]);
    for(const p of [start,anchor,drag]){dot(p,3,'255,255,255',1);cross(p,5,.6);}
    if(phase.step===0){microCursor(start,.6);}
    else if(phase.step===1){microCursor(onCurve(path,smooth(phase.p)),.95);}
    else if(phase.step===2){
      const cursor=pointMix(anchor,drag,smooth(phase.p));microCursor(cursor,.95);
      corners(anchor.x-14,anchor.y-14,28,28,cyan,.9,6);
      line([anchor,cursor],cyan,.95,1.7);bloom(cursor,28,cyan,.2);
    } else {
      microCursor(drag,.8);
      ctx.beginPath();ctx.arc(anchor.x,anchor.y,21,0,TAU*phase.feedback);ctx.strokeStyle=`rgba(${cyan},.75)`;ctx.lineWidth=1.4;ctx.stroke();
      line([{x:anchor.x-5,y:anchor.y},{x:anchor.x-1,y:anchor.y+5},{x:anchor.x+7,y:anchor.y-6}],cyan,.95,1.7);
    }
  }

  function interaction() {
    if(pointer.strength<.02||reduced.matches)return;
    // The real mouse produces only a soft local light, with no reticle or coordinates.
    bloom(pointer,46,cyan,.16*pointer.strength);
    bloom(pointer,13,'255,255,255',.58*pointer.strength);
  }

  function render() {
    if(!width||!height)return;
    ctx.clearRect(0,0,width,height);
    const phase=phases();
    drawingField();coordinatePlane(phase);
    const anchor=scientificSurface(phase);
    perceptionRaster(phase);
    molecularModels();celestialOrbits();geospatialMap();proofEvolution();typesetting();
    operation(anchor,phase);interaction();
    ctx.save();ctx.globalCompositeOperation='destination-out';ctx.drawImage(mask,0,0,width,height);ctx.restore();
    canvas.dataset.effect='agent-vision';canvas.dataset.phase=['observe','locate','act','verify'][phase.step];
    canvas.dataset.models=width>=900?'kalgebra,chimerax,celestia,grass,lean,texstudio':width>=700?'kalgebra,chimerax,celestia,grass':'kalgebra';
    canvas.dataset.interaction=pointer.active?'engaged':'idle';
  }
  function tick(now) {
    raf=null;if(paused||!visible||document.hidden){last=0;return;}
    if(!last)last=now;
    const delta=now-last;
    if(delta>=32){
      time+=Math.min(delta,90);last=now;
      pointer.strength+=(Number(pointer.active)-pointer.strength)*.14;
      render();
    }
    raf=requestAnimationFrame(tick);
  }
  function start(){if(raf===null&&!paused&&visible&&!document.hidden)raf=requestAnimationFrame(tick);}
  function pause(value){
    paused=value;
    if(paused&&raf!==null){cancelAnimationFrame(raf);raf=null;last=0;}
    start();
  }
  document.addEventListener('pointermove',event=>{
    if(event.pointerType!=='mouse'&&event.pointerType!=='pen')return;
    const r=canvas.getBoundingClientRect();pointer.active=event.clientY>=r.top&&event.clientY<=r.bottom&&!event.target.closest('dialog');
    if(pointer.active){pointer.x=event.clientX-r.left;pointer.y=event.clientY-r.top;}start();
  },{passive:true});
  document.documentElement.addEventListener('pointerleave',()=>{pointer.active=false;},{passive:true});
  document.addEventListener('visibilitychange',()=>{last=0;start();});
  new IntersectionObserver(entries=>{visible=entries[0].isIntersecting;last=0;start();},{threshold:.01}).observe(hero);
  new ResizeObserver(fit).observe(hero);window.addEventListener('resize',fit,{passive:true});document.fonts.ready.then(fit);
  reduced.addEventListener('change',()=>pause(reduced.matches));fit();pause(paused);
})();
