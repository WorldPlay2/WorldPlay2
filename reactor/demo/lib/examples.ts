export type Perspective = "fps" | "tps";

/** A named event: one sentence added to the prompt while its chip or hotkey is held. */
export type ExampleEvent = {
  name: string;
  text: string;
};

/** A starting image plus the prompt parts and perspective it is played with. */
export type Example = {
  id: string;
  name: string;
  perspective: Perspective;
  /** What "The scene in the video is:" describes. */
  scene: string;
  /** What "The character is:" describes; used only for tps. */
  character: string;
  events: ExampleEvent[];
  /** URL of the image: a file under public/ or an object URL for a saved example. */
  imageSrc: string;
};

/** Hotkeys 1–9 address the first nine events. */
export const MAX_EVENTS = 9;

const DEFAULT_SCENE =
  "the environment shown in the reference image, with consistent layout, objects, and lighting";
const DEFAULT_CHARACTER =
  "the same main character shown in the reference image, with consistent appearance and clothing";

function sentencePart(text: string): string {
  return text.trim().replace(/[.\s]+$/, "");
}

/** The base prompt: scene, plus character for tps. Blank parts use the model's own defaults. */
export function composeBasePrompt(parts: Pick<Example, "perspective" | "scene" | "character">): string {
  const scene = sentencePart(parts.scene) || DEFAULT_SCENE;
  let prompt = `The scene in the video is: ${scene}.`;
  if (parts.perspective === "tps") {
    prompt += ` The character is: ${sentencePart(parts.character) || DEFAULT_CHARACTER}.`;
  }
  return prompt;
}

/** Appends "The event is: ..." to a base prompt; returns the base unchanged without an event. */
export function withEvent(base: string, event: ExampleEvent | null): string {
  const text = event ? sentencePart(event.text) : "";
  return text ? `${base} The event is: ${text}.` : base;
}

/** Drops any "The event is: ..." sentence, leaving the base prompt. */
export function stripEvent(prompt: string): string {
  return prompt.replace(/\s*The event is:[\s\S]*$/, "").trim();
}

/** Splits a full prompt back into scene and character parts. */
export function splitPrompt(prompt: string): { scene: string; character: string } {
  const base = stripEvent(prompt);
  const match = base.match(/^\s*The scene in the video is:\s*([\s\S]*?)(?:\s*The character is:\s*([\s\S]*))?$/);
  if (!match) return { scene: base, character: "" };
  return { scene: sentencePart(match[1] ?? ""), character: sentencePart(match[2] ?? "") };
}

export const BUILT_IN_EXAMPLES: Example[] = [
  {
    id: "example-01",
    name: "Noir Alley Patrol",
    perspective: "tps",
    imageSrc: "/examples/example-01.jpg",
    scene: "A lone uniformed police officer in dark blue tactical gear in a narrow urban alley at night. The world contains EXACTLY ONE tall street lamp on the right at a fixed position AND EXACTLY ONE glowing neon shop sign on the left at a fixed position AND EXACTLY ONE shop door straight ahead at a fixed position AND EXACTLY ONE green dumpster on the right at a fixed position. Dark brick walls, heavy rain falling, shiny puddles on the wet asphalt, yellow police tape, blue and red ambient light. Cinematic noir night, reflective wet surfaces.",
    character: "A lone uniformed police officer in dark blue tactical gear",
    events: [
      {
        name: "Fire Pistol",
        text: "The officer raises his service pistol in both gloved hands, arms extended ahead of him, and fires down the dark alley; the muzzle flash lights the falling rain, the recoil kicks the pistol back in his grip, and a spent casing clatters onto the wet asphalt.",
      },
      {
        name: "Flamethrower",
        text: "The officer levels a flamethrower in both hands and unleashes a long, continuous jet of roaring flame straight ahead down the alley — one sustained, billowing stream of orange fire and black smoke that pours forward without pause, not separate shots, lighting up the wet brick walls and hissing against the falling rain.",
      },
      {
        name: "Fire Rocket Launcher",
        text: "The officer hoists a shoulder-fired rocket launcher — an RPG: a long, heavy cylindrical steel tube resting across his shoulder, its wide bore pointing down the alley, gripped with both hands, not a handheld gun — and fires; the rocket blasts from the tube's mouth and flies off into the distance down the rain-slicked alley, dragging a long tail of orange flame and white smoke behind it, then slams into the distant brickwork and erupts in a massive fireball of flame, smoke, and flying debris.",
      },
      {
        name: "Knife Attack",
        text: "The officer draws a combat knife into his gloved hand and slashes at the air in a swift, practiced motion; the steel blade flashes in the ambient light as it cuts through the rain and his grip tightens on the handle.",
      },
      {
        name: "Forward Roll",
        text: "The officer drops into a low crouch and executes a forward roll across the wet asphalt, tucking his shoulder and rising smoothly back to his feet.",
      },
    ],
  },
  {
    id: "example-02",
    name: "Battlefield Horseman",
    perspective: "tps",
    imageSrc: "/examples/example-02.jpg",
    scene: "A warrior in green armor and a hood, mounted on a brown horse and holding a large curved blade, on a muddy battlefield strewn with scattered debris and a distant burning campfire. Somber, war-torn atmosphere under a heavy grey sky.",
    character: "A warrior in green armor and a hood, mounted on a brown horse and holding a large curved blade",
    events: [
      {
        name: "Snowstorm Approaches",
        text: "The sky darkens rapidly and a heavy blizzard sweeps in across the battlefield, howling wind driving dense sheets of thick snow sideways through the air. The whirling snowfall quickly builds into a near-whiteout, snow piling up over the muddy terrain, the warrior's armor and hood, the horse's back, and the scattered debris, frost creeping across every surface as the war-torn field is swallowed by a cold, roaring, white-grey storm.",
      },
      {
        name: "Horse Lunge Strike",
        text: "The warrior leans forward, guiding the horse into a sudden lunge. The large curved blade swings in a wide, downward arc, cleaving through the mud and shattering a nearby wooden shield into splinters.",
      },
      {
        name: "Casts Fire Magic",
        text: "The warrior raises the large curved blade high overhead toward the sky, and dark storm-clouds gather above as the blade glows with channeled energy. In answer, a rain of blazing fireballs streaks down from the sky, slamming into the muddy battlefield one after another in bursts of fire, smoke, and flying debris.",
      },
    ],
  },
  {
    id: "example-03",
    name: "Jet Ski Cruise",
    perspective: "tps",
    imageSrc: "/examples/example-03.jpg",
    scene: "A man in a red life vest on a white and red jet ski on turquoise water near a sandy beach lined with palm trees, a distant rocky outcrop on the horizon. Sunlit coastal atmosphere with light glinting off the calm sea.",
    character: "A man in a red life vest on a white and red jet ski",
    events: [
      {
        name: "Meteor Shower",
        text: "Single meteors fall from high in the sky one after another — each a giant burning ball of rock dragging a long tail of fire and smoke straight down toward the sea. One by one they crash into the water far off near the horizon, away from the rider and never on the path ahead, each erupting in a huge fiery explosion and a towering burst of white spray before the next one follows.",
      },
      {
        name: "Sunset Glow",
        text: "The daylight deepens into a fiery sunset: the sun sinks low and rests right on the horizon, a huge, swollen orange-red disc half-dipping into the sea, while the sky above ignites in deep bands of amber, rose, and violet. Directly beneath it, the sun's reflection blazes across the water — a brilliant, unbroken path of molten gold stretching from the horizon straight toward the viewer, the rippling sea breaking it into shimmering streaks on every wavelet, the beach glowing warm amber.",
      },
      {
        name: "Rides with Torch",
        text: "The rider raises a burning torch high overhead in one hand, its orange flame streaming and flickering in the wind and trailing a ribbon of smoke, while his other hand grips the handlebars and the jet ski drives forward across the water, spray fanning off the hull.",
      },
    ],
  },
  {
    id: "example-04",
    name: "Dragon Flying",
    perspective: "fps",
    imageSrc: "/examples/example-04.jpg",
    scene: "This is a first-person-view video of a colossal dragon — its neck a column of dense obsidian-black scales rippling with muscle, its enormous bat-like wings veined with pulsing crimson, leading edges sharp and dark. Ahead, a towering ancient castle of crumbling stone spires and moss-covered battlements rises above dense jungle canopy, gothic arches half-swallowed by creeping vines. A vast primordial forest stretches in every direction, pale mist drifting between trunks, dappled golden sunlight filtering through humid air, winding river gorges far below.",
    character: "",
    events: [
      {
        name: "Portal",
        text: "As a massive burst of brilliant fireworks erupts across the entire sky — a dazzling explosion of red, gold, violet, and emerald green light blooming in spectacular fashion, the radiant blast filling the screen completely with cascading sparks and shimmering trails. At the peak of the explosion, the fireworks coalesce into a luminous tear in the fabric of reality, a glowing fissure splitting open through the heart of the brilliance, crackling with chromatic energy along its jagged edges. As the fireworks disperse and the sparks fade away in spiraling ribbons of smoke, the dragon emerges through the closing rift to find itself before a sprawling expanse of ancient castles — towering stone spires, weathered battlements, and gothic arches stretching across the horizon, their silhouettes rising majestically against a new sky. The colorful afterglow of the dimensional passage still dances across the dragon's enormous outstretched wings and glints off its obsidian-black armored scales, faint embers from the dissipating rift lingering in the air around the beast as it soars toward the castle landscape ahead.",
      },
      {
        name: "Fire breath",
        text: "Further ahead, the dragon's head is clearly defined and held level with its body, neck stretched forward, jaws wide open and aimed forward at the horizon — the head is NOT tilted up toward the sky — positioned within the middle distance, not fully filling the frame. From this open mouth, a tightly collimated cylindrical column of brilliant fire shoots horizontally forward along the dragon's flight axis, parallel to the ground far below — NOT angled upward into the sky — projecting straight into the far distance like a focused beam, a narrow cohesive pillar of flame that maintains a consistent diameter along its entire length, NOT a fan, cone, or diffuse spray. The fire column is white-hot at the core with blazing orange and gold edges. The dragon's obsidian-black scales glow molten-red along its neck and back from the sheer radiance. Sparks and glowing embers cascade outward. A dense forest far below is bathed in intense flickering firelight.",
      },
    ],
  },
];
