"use strict";

// Prepare scene
let scene = new THREE.Scene();
let camera = new THREE.OrthographicCamera(-5000, 5000, 5000, -5000, -5000, 5000);
camera.position.z = 2000;
let renderer = new THREE.WebGLRenderer();
renderer.setSize(window.innerWidth, window.innerHeight);
document.body.appendChild(renderer.domElement);

// Add planets
let planetsData = {
  planet1: {
    fixed: true,
    radius: 100,
    mass: 10000000,
    color: 0xffe030,
    position: {
      x: 0,
      y: 0,
      z: 0
    },
    velocity: {
      x: 0,
      y: 0,
      z: 0
    },
    acceleration: {
      x: 0,
      y: 0,
      z: 0
    }
  },
  planet2: {
    radius: 20,
    mass: 20,
    color: 0xff5533,
    position: {
      x: 800,
      y: 0,
      z: 500
    },
    velocity: {
      x: 0,
      y: 110,
      z: 0
    },
    acceleration: {
      x: 0,
      y: 0,
      z: 0
    }
  },
  planet3: {
    radius: 20,
    mass: 10,
    color: 0x80ff80,
    position: {
      x: 0,
      y: 1000,
      z: 500
    },
    velocity: {
      x: -110,
      y: 20,
      z: 10
    },
    acceleration: {
      x: 0,
      y: 0,
      z: 0
    }
  },
  planet4: {
    radius: 20,
    mass: 10,
    color: 0x3399ff,
    position: {
      x: -800,
      y: -600,
      z: -300
    },
    velocity: {
      x: 50,
      y: -70,
      z: 0
    },
    acceleration: {
      x: 0,
      y: 0,
      z: 0
    }
  },
  planet5: {
    radius: 30,
    mass: 15,
    color: 0x88bbff,
    position: {
      x: 800,
      y: -1200,
      z: -100
    },
    velocity: {
      x: 50,
      y: 70,
      z: 0
    },
    acceleration: {
      x: 0,
      y: 0,
      z: 0
    }
  },
  planet6: {
    radius: 50,
    mass: 20,
    color: 0xcc00ff,
    position: {
      x: 500,
      y: 500,
      z: -50
    },
    velocity: {
      x: -100,
      y: 80,
      z: 0
    },
    acceleration: {
      x: 0,
      y: 0,
      z: 0
    }
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
  setTimeout(function() {
    requestAnimationFrame(animate);
  }, 1000/120);

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
      let r = Math.sqrt(distanceX**2 + distanceY**2 + distanceZ**2);

      // Calculate acting forces
      let acceleration = refPlanet.data.mass / r**2;
      let accelerationX = acceleration * distanceX / r;
      let accelerationY = acceleration * distanceY / r;
      let accelerationZ = acceleration * distanceZ / r;

      // Set acting forces.
      planet.data.acceleration.x -= accelerationX;
      planet.data.acceleration.y -= accelerationY;
      planet.data.acceleration.z -= accelerationZ;

      acceleration = planet.data.mass / r**2;
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
        // console.log(planet);
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

  let time = 0.2;

  for (let planet of planets) {
    // Update velocities
    planet.data.velocity.x += time * planet.data.acceleration.x;
    planet.data.velocity.y += time * planet.data.acceleration.y;
    planet.data.velocity.z += time * planet.data.acceleration.z;

    // Update displacements
    planet.data.position.x += time * planet.data.velocity.x;
    planet.data.position.y += time * planet.data.velocity.y;
    planet.data.position.z += time * planet.data.velocity.z;

    // Update positions drawn
    planet.position.x = planet.data.position.x;
    planet.position.y = planet.data.position.y;
    planet.position.z = planet.data.position.z;

    for (let i = 69; i > 0; i--) {
      planet.trails[i].position.x = planet.trails[i - 1].position.x;
      planet.trails[i].position.y = planet.trails[i - 1].position.y;
      planet.trails[i].position.z = planet.trails[i - 1].position.z;
    }
    planet.trails[0].position.x = planet.position.x;
    planet.trails[0].position.y = planet.position.y;
    planet.trails[0].position.z = planet.position.z;
  }
}
