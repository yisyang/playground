"use strict";

let piano = Synth.createInstrument('piano');
let lastPlayedSound = -Infinity;

// Prepare scene
let scene = new THREE.Scene();
let camera = new THREE.OrthographicCamera(-6000, 6000, 5600, -6000, -6000, 6000);
camera.position.z = 2000;
let renderer = new THREE.WebGLRenderer();
renderer.setSize(Math.min(window.innerWidth, window.innerHeight), Math.min(window.innerWidth, window.innerHeight));
document.body.appendChild(renderer.domElement);

// Add planets
let planetsData = {
  planet1: {
    fixed: true,
    radius: 100,
    mass: 10000000,
    color: 0xffe030,
    position: { x: 0, y: 0, z: 0 },
    velocity: { x: 0, y: 0, z: 0 },
    acceleration: { x: 0, y: 0, z: 0 },
    note: 'C',
    pitch: 2
  },
  planet2: {
    radius: 50,
    mass: 200,
    color: 0xff0000,
    position: { x: 800, y: 0, z: 0 },
    velocity: { x: 0, y: 110, z: 0 },
    acceleration: { x: 0, y: 0, z: 0 },
    note: 'C',
    pitch: 3
  },
  planet3: {
    radius: 50,
    mass: 200,
    color: 0xff5580,
    position: { x: -800, y: 0, z: 0 },
    velocity: { x: 0, y: -110, z: 0 },
    acceleration: { x: 0, y: 0, z: 0 },
    note: 'D',
    pitch: 3
  },
  planet4: {
    radius: 50,
    mass: 200,
    color: 0xffaa00,
    position: { x: 0, y: 1000, z: 0 },
    velocity: { x: -110, y: 20, z: 0 },
    acceleration: { x: 0, y: 0, z: 0 },
    note: 'E',
    pitch: 3
  },
  planet5: {
    radius: 50,
    mass: 200,
    color: 0xffff00,
    position: { x: 0, y: -1000, z: 0 },
    velocity: { x: 110, y: -20, z: 0 },
    acceleration: { x: 0, y: 0, z: 0 },
    note: 'F',
    pitch: 3
  },
  planet6: {
    radius: 50,
    mass: 200,
    color: 0xaaff00,
    position: { x: 500, y: 500, z: 0 },
    velocity: { x: -100, y: 80, z: 0 },
    acceleration: { x: 0, y: 0, z: 0 },
    note: 'G',
    pitch: 3
  },
  planet7: {
    radius: 50,
    mass: 200,
    color: 0x55ff00,
    position: { x: -500, y: -500, z: 0 },
    velocity: { x: 100, y: -80, z: 0 },
    acceleration: { x: 0, y: 0, z: 0 },
    note: 'A',
    pitch: 3
  },
  planet8: {
    radius: 50,
    mass: 200,
    color: 0x00ff55,
    position: { x: 500, y: 200, z: 0 },
    velocity: { x: -80, y: 80, z: 0 },
    acceleration: { x: 0, y: 0, z: 0 },
    note: 'B',
    pitch: 3
  },
  planet9: {
    radius: 30,
    mass: 100,
    color: 0x00ffff,
    position: { x: -500, y: -200, z: 0 },
    velocity: { x: 80, y: -80, z: 0 },
    acceleration: { x: 0, y: 0, z: 0 },
    note: 'C',
    pitch: 4
  },
  planet10: {
    radius: 30,
    mass: 100,
    color: 0x00aaff,
    position: { x: 200, y: 550, z: 0 },
    velocity: { x: -100, y: 30, z: 0 },
    acceleration: { x: 0, y: 0, z: 0 },
    note: 'D',
    pitch: 4
  },
  planet11: {
    radius: 30,
    mass: 100,
    color: 0x0055ff,
    position: { x: -250, y: -500, z: 0 },
    velocity: { x: 100, y: -30, z: 0 },
    acceleration: { x: 0, y: 0, z: 0 },
    note: 'E',
    pitch: 4
  },
  planet12: {
    radius: 30,
    mass: 100,
    color: 0x0000ff,
    position: { x: 850, y: 600, z: 0 },
    velocity: { x: -50, y: 70, z: 0 },
    acceleration: { x: 0, y: 0, z: 0 },
    note: 'F',
    pitch: 4
  },
  planet13: {
    radius: 30,
    mass: 100,
    color: 0x5500ff,
    position: { x: -800, y: -650, z: 0 },
    velocity: { x: 50, y: -70, z: 0 },
    acceleration: { x: 0, y: 0, z: 0 },
    note: 'G',
    pitch: 4
  },
  planet14: {
    radius: 30,
    mass: 100,
    color: 0xaa00ff,
    position: { x: 900, y: -1200, z: 0 },
    velocity: { x: 50, y: 70, z: 0 },
    acceleration: { x: 0, y: 0, z: 0 },
    note: 'A',
    pitch: 4
  },
  planet15: {
    radius: 30,
    mass: 100,
    color: 0xff00aa,
    position: { x: -800, y: 1100, z: 0 },
    velocity: { x: -50, y: -70, z: 0 },
    acceleration: { x: 0, y: 0, z: 0 },
    note: 'B',
    pitch: 4
  }
};
let planets = [];
for (let planetData of Object.values(planetsData)) {
  planets = addPlanet(planetData, planets);
}

animate(planets);

function addPlanet(planetData, planets) {
  let geometry = new THREE.SphereGeometry(planetData.radius);
  let material = new THREE.MeshBasicMaterial({ color: planetData.color });

  let planet = new THREE.Mesh(geometry, material);
  planet.data = planetData;
  planet.position.x = planetData.position.x || 0;
  planet.position.y = planetData.position.y || 0;
  planet.position.z = planetData.position.z || 0;
  planets.push(planet);
  scene.add(planet);

  planet.trails = [];
  for (let i = 0; i < 70; i++) {
    geometry = new THREE.SphereGeometry(planetData.radius * (1 - 0.01 * i));
    material = new THREE.MeshBasicMaterial({ color: lightenColor(planetData.color, i) });
    let trailPlanet = new THREE.Mesh(geometry, material);
    trailPlanet.position.x = planetData.position.x || 0;
    trailPlanet.position.y = planetData.position.y || 0;
    trailPlanet.position.z = planetData.position.z || 0;
    planet.trails[i] = trailPlanet;
    scene.add(trailPlanet);
  }

  return planets;
}

function animate() {
  // Throttle to max 120 FPS
  setTimeout(function () {
    requestAnimationFrame(animate);
  }, 1000 / 120);

  movePlanets(planets);

  renderer.render(scene, camera);
}

function lightenColor(color, percent) {
  let num = color,
    amt = Math.round(2.55 * percent),
    R = (num >> 16) + amt,
    B = (num >> 8 & 0x00FF) + amt,
    G = (num & 0x0000FF) + amt;

  return (0x1000000 + (R < 255 ? R < 1 ? 0 : R : 255) * 0x10000 + (B < 255 ? B < 1 ? 0 : B : 255) * 0x100 + (G < 255 ? G < 1 ? 0 : G : 255));
}

function movePlanets(planets) {
  for (let planet of planets) {
    planet.data.forceCalculated = false;
    planet.data.acceleration = {
      x: 0,
      y: 0,
      z: 0
    };
  }

  for (let planet of planets) {
    for (let refPlanet of planets) {
      if (planet.uuid === refPlanet.uuid) {
        continue;
      }
      if (refPlanet.data.forceCalculated) {
        continue;
      }

      let distanceX = planet.data.position.x - refPlanet.data.position.x;
      let distanceY = planet.data.position.y - refPlanet.data.position.y;
      let distanceZ = planet.data.position.z - refPlanet.data.position.z;
      let r = Math.sqrt(distanceX ** 2 + distanceY ** 2 + distanceZ ** 2);

      // Calculate acting forces
      let acceleration = refPlanet.data.mass / r ** 2;
      let accelerationX = acceleration * distanceX / r;
      let accelerationY = acceleration * distanceY / r;
      let accelerationZ = acceleration * distanceZ / r;

      // Set acting forces.
      planet.data.acceleration.x -= accelerationX;
      planet.data.acceleration.y -= accelerationY;
      planet.data.acceleration.z -= accelerationZ;

      acceleration = planet.data.mass / r ** 2;
      accelerationX = acceleration * distanceX / r;
      accelerationY = acceleration * distanceY / r;
      accelerationZ = acceleration * distanceZ / r;

      // Set opposing forces.
      refPlanet.data.acceleration.x += accelerationX;
      refPlanet.data.acceleration.y += accelerationY;
      refPlanet.data.acceleration.z += accelerationZ;
    }
    planet.data.forceCalculated = true;
  }

  for (let planet of planets) {
    if (planet.data.fixed) {
      for (let refPlanet of planets) {
        if (planet.uuid === refPlanet.uuid) {
          continue;
        }
        if (refPlanet.data.forceCalculated) {
          refPlanet.data.acceleration.x -= planet.data.acceleration.x;
          refPlanet.data.acceleration.y -= planet.data.acceleration.y;
          refPlanet.data.acceleration.z -= planet.data.acceleration.z;
        }
      }
      planet.data.acceleration.x = 0;
      planet.data.acceleration.y = 0;
      planet.data.acceleration.z = 0;
      break;
    }
  }

  let time = 1;

  for (let planet of planets) {
    // Update velocities
    planet.data.velocity.x += time * planet.data.acceleration.x;
    planet.data.velocity.y += time * planet.data.acceleration.y;
    planet.data.velocity.z += time * planet.data.acceleration.z;

    // Hack - dampen velocities that are wayyy too high
    if (planet.data.velocity.x > 500) {
      planet.data.velocity.x = 500;
    }
    if (planet.data.velocity.x < -500) {
      planet.data.velocity.x = -500;
    }
    if (planet.data.velocity.y > 500) {
      planet.data.velocity.y = 500;
    }
    if (planet.data.velocity.y < -500) {
      planet.data.velocity.y = -500;
    }
    if (planet.data.velocity.z > 500) {
      planet.data.velocity.z = 500;
    }
    if (planet.data.velocity.z < -500) {
      planet.data.velocity.z = -500;
    }

    // Update displacements
    planet.data.position.x += time * planet.data.velocity.x;
    planet.data.position.y += time * planet.data.velocity.y;
    planet.data.position.z += time * planet.data.velocity.z;

    // Update positions drawn
    planet.position.x = planet.data.position.x;
    planet.position.y = planet.data.position.y;
    planet.position.z = planet.data.position.z;

    // Hack - if something went out too far, dampen their velocities
    if (planet.position.x > 5000 && planet.data.velocity.x > 0) {
      planet.data.velocity.x = 0.1 * planet.data.velocity.x;
    }
    if (planet.position.y > 5000 && planet.data.velocity.y > 0) {
      planet.data.velocity.y = 0.1 * planet.data.velocity.y;
    }
    if (planet.position.z > 5000 && planet.data.velocity.z > 0) {
      planet.data.velocity.z = 0.1 * planet.data.velocity.z;
    }
    if (planet.position.x < -5000 && planet.data.velocity.x < 0) {
      planet.data.velocity.x = 0.1 * planet.data.velocity.x;
    }
    if (planet.position.y < -5000 && planet.data.velocity.y < 0) {
      planet.data.velocity.y = 0.1 * planet.data.velocity.y;
    }
    if (planet.position.z < -5000 && planet.data.velocity.z < 0) {
      planet.data.velocity.z = 0.1 * planet.data.velocity.z;
    }

    // If previous position is at further away than the next previous position
    // and the current position, we passed maxima.
    let currentDistanceSquared = planet.position.x ** 2 + planet.position.y ** 2;
    let previousDistanceSquared = planet.trails[0].position.x ** 2 + planet.trails[0].position.y ** 2;
    let previousPreviousDistanceSquared = planet.trails[1].position.x ** 2 + planet.trails[1].position.y ** 2;
    if (previousDistanceSquared > currentDistanceSquared && previousDistanceSquared > previousPreviousDistanceSquared) {
      let now = (new Date() | 0);
      let elapsed = now - lastPlayedSound;
      // After passing maxima, play sound with a chance.
      if (elapsed > 2000
        || (elapsed > 500 && now % 2 && now % 3)
      ) {
        lastPlayedSound = now;
        piano.play(planet.data.note, planet.data.pitch, 2);
        console.log(planet.trails[0].geometry);
        planet.trails[0].geometry.parameters.radius = 500;
        planet.trails[0].geometry.boundingSphere.radius = 500;
        planet.trails[0].scale.x = 15;
        planet.trails[0].scale.y = 15;
      }
    }

    for (let i = 69; i > 0; i--) {
      planet.trails[i].position.x = planet.trails[i - 1].position.x;
      planet.trails[i].position.y = planet.trails[i - 1].position.y;
      planet.trails[i].position.z = planet.trails[i - 1].position.z;
      planet.trails[i].scale.x = planet.trails[i - 1].scale.x;
      planet.trails[i].scale.y = planet.trails[i - 1].scale.y;
    }
    planet.trails[0].position.x = planet.position.x;
    planet.trails[0].position.y = planet.position.y;
    planet.trails[0].position.z = planet.position.z;
    planet.trails[0].scale.x = 1;
    planet.trails[0].scale.y = 1;
  }
}
