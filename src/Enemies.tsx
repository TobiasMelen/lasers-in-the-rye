import { useFrame } from "@react-three/fiber";
import { useEffect, useMemo } from "react";
import * as THREE from "three/webgpu";
import { ISO_RIGHT, ISO_UP } from "./iso";
import { Bot, useBotAssets } from "./bot";
import type { BoltPool, Target } from "./bolts";
import type { Stains } from "./stains";

// Gold enemy bots: they appear past the top of the screen, walk down
// towards the player, then hold at range, strafing and shooting. One player
// bolt blows them apart (the bolt pool draws the burst), leaving blood on
// the grass, and they go back to the pool

const MAX_ENEMIES = 4;
const SPAWN_INTERVAL = 2.5; // seconds between arrivals while below the max
// Where they appear: this far up the screen's vertical from the player
// (past the top edge even with the camera leading upwards), spread across
const SPAWN_DISTANCE = 34; // metres along the ground
const SPAWN_SPREAD = 14; // metres either side
const DESPAWN_DISTANCE = 60; // wandered this far off, gone

const APPROACH_SPEED = 3.2; // metres per second
const STRAFE_SPEED = 1.6;
const ACCELERATION = 4; // how quickly velocity reaches the intent, per second
const ATTACK_RANGE = 12; // holds about this far from the player
const FIRE_RANGE = 16; // starts shooting from this close
const FIRE_INTERVAL: [number, number] = [0.9, 1.6]; // seconds, random between
const AIM_ERROR = 0.07; // radians either side
const PERSONAL_SPACE = 2.2; // metres kept between enemies

const HIT_RADIUS = 0.6;

const up = new THREE.Vector3(ISO_UP[0], 0, ISO_UP[1]);
const right = new THREE.Vector3(ISO_RIGHT[0], 0, ISO_RIGHT[1]);
const toPlayer = new THREE.Vector3();
const intent = new THREE.Vector3();
const away = new THREE.Vector3();
const shot = new THREE.Vector3();

type Enemy = {
  bot: Bot;
  target: Target;
  active: boolean;
  cooldown: number;
  strafe: number; // -1 or 1, and when to reconsider it
  strafeTimer: number;
};

const random = (min: number, max: number) => min + Math.random() * (max - min);

export default function Enemies({
  player,
  bolts,
  grassPushers,
  stains,
  active,
}: {
  player: THREE.Vector3;
  bolts: BoltPool;
  grassPushers: Set<THREE.Vector3>;
  stains: Stains;
  // Hold off (no arrivals, nothing moves) until play starts
  active: boolean;
}) {
  const assets = useBotAssets();

  // A fixed pool, built up front so arrivals don't stall a frame
  const { group, enemies, leave, spawn } = useMemo(() => {
    const group = new THREE.Group();
    const enemies: Enemy[] = [];
    const leave = (enemy: Enemy) => {
      enemy.active = false;
      enemy.bot.object.visible = false;
      bolts.targets.delete(enemy.target);
      grassPushers.delete(enemy.bot.position);
    };
    for (let i = 0; i < MAX_ENEMIES; i++) {
      const bot = new Bot(assets, "gold");
      bot.object.visible = false;
      group.add(bot.object);
      const enemy: Enemy = {
        bot,
        target: {
          team: "enemy",
          position: bot.position,
          radius: HIT_RADIUS,
          explodes: true,
          hit: (direction) => {
            stains.add(bot.position.x, bot.position.z, direction);
            leave(enemy);
          },
        },
        active: false,
        cooldown: 0,
        strafe: 1,
        strafeTimer: 0,
      };
      enemies.push(enemy);
    }
    return { group, enemies, leave, spawn: { timer: 1 } };
  }, [assets, bolts, grassPushers, stains]);
  useEffect(
    () => () => {
      for (const enemy of enemies) {
        bolts.targets.delete(enemy.target);
        grassPushers.delete(enemy.bot.position);
      }
    },
    [enemies, bolts, grassPushers]
  );

  useFrame((_, delta) => {
    if (!active) return;
    // Arrivals
    spawn.timer -= delta;
    if (spawn.timer <= 0) {
      spawn.timer = SPAWN_INTERVAL;
      const enemy = enemies.find((e) => !e.active);
      if (enemy) {
        const { bot } = enemy;
        bot.velocity.set(0, 0, 0);
        bot.position
          .copy(player)
          .addScaledVector(up, SPAWN_DISTANCE)
          .addScaledVector(right, random(-SPAWN_SPREAD, SPAWN_SPREAD));
        bot.face(toPlayer.subVectors(player, bot.position));
        bot.object.visible = true;
        enemy.active = true;
        enemy.cooldown = random(...FIRE_INTERVAL);
        enemy.strafe = Math.random() < 0.5 ? -1 : 1;
        enemy.strafeTimer = random(1.5, 3);
        bolts.targets.add(enemy.target);
        grassPushers.add(bot.position);
      }
    }

    for (const enemy of enemies) {
      if (!enemy.active) continue;
      const { bot } = enemy;
      toPlayer.subVectors(player, bot.position).setY(0);
      const distance = toPlayer.length();
      toPlayer.divideScalar(Math.max(distance, 1e-3));

      if (distance > DESPAWN_DISTANCE) {
        leave(enemy);
        continue;
      }

      // Close in until at range, then circle the player, easing in or out
      // to hold that range
      enemy.strafeTimer -= delta;
      if (enemy.strafeTimer <= 0) {
        enemy.strafe = -enemy.strafe;
        enemy.strafeTimer = random(1.5, 3);
      }
      if (distance > ATTACK_RANGE + 2) {
        intent.copy(toPlayer).multiplyScalar(APPROACH_SPEED);
      } else {
        intent
          .set(-toPlayer.z, 0, toPlayer.x)
          .multiplyScalar(enemy.strafe * STRAFE_SPEED)
          .addScaledVector(toPlayer, THREE.MathUtils.clamp(distance - ATTACK_RANGE, -1, 1) * STRAFE_SPEED);
      }
      // Keep apart from the others
      for (const other of enemies) {
        if (other === enemy || !other.active) continue;
        away.subVectors(bot.position, other.bot.position).setY(0);
        const gap = away.length();
        if (gap < PERSONAL_SPACE && gap > 1e-3) {
          intent.addScaledVector(away, ((PERSONAL_SPACE - gap) / gap) * 2);
        }
      }
      bot.velocity.lerp(intent, 1 - Math.exp(-delta * ACCELERATION));
      bot.update(delta, toPlayer);

      // Shoot at the player, not quite dead on
      enemy.cooldown -= delta;
      if (enemy.cooldown <= 0 && distance < FIRE_RANGE) {
        enemy.cooldown = random(...FIRE_INTERVAL);
        shot.copy(player).sub(bot.muzzle).setY(0).normalize();
        shot.applyAxisAngle(THREE.Object3D.DEFAULT_UP, random(-AIM_ERROR, AIM_ERROR));
        bolts.fire("enemy", bot.muzzle, shot);
        bot.recoil();
      }
    }
  });

  return <primitive object={group} />;
}
