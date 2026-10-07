# WorldPlay2 frontend

A browser client for WorldPlay2: pick an image and a prompt, then explore the generated world with the keyboard.

## Setup

```sh
cp .env.example .env
pnpm install
pnpm dev
```

Open http://localhost:3000, choose an endpoint and click **Connect**.

The endpoint list comes from `endpoints.json`, which ships with a local entry:

- **Local** (default) connects to the model started with `reactor run` at `http://localhost:8080`. The URL can also be edited in the page (remembered per endpoint in this browser). No API key is needed.

The demo also supports optional hosted endpoints. Add one as described below and set
its `apiKeyEnv` variable in `.env`; it is read only by the server route `app/api/token`
and never sent to the browser. No `.env` or API key is needed for local use.

### Add an endpoint

Create `endpoints.local.json` next to `endpoints.json` (it is gitignored, so your hosts stay on your machine) and reload the page. Its entries are added to the list; an entry with the same `id` as a shipped one replaces it.

```json
{
  "endpoints": [
    { "label": "GPU box", "url": "http://<gpu-host>:8080", "kind": "local" },
    { "label": "Other API", "url": "https://<api-host>", "kind": "hosted", "apiKeyEnv": "MY_API_KEY" }
  ]
}
```

| Field | Meaning |
| --- | --- |
| `label` | Name shown in the endpoint menu |
| `url` | `local`: the model runtime's address. `hosted`: the Reactor API base URL |
| `kind` | `local` talks to the model runtime directly, no key. `hosted` gets a session token from `app/api/token` |
| `apiKeyEnv` | `hosted` only: the `.env` variable holding that API's key. The server reads the URL and the key from these files; the browser sends only the endpoint id |
| `id` | Optional; defaults to the label in lowercase-with-dashes. Must be unique |
| `modelName` | Optional, `hosted` only: the model's name on that API when it is not `worldplay2`, e.g. a canonical `org/worldplay2` |

## Controls

| Input | Effect |
| --- | --- |
| `W` `A` `S` `D` | Move forward, left, backward, right; two keys combine into a diagonal |
| `←` `→` | Turn left / right |
| `↑` `↓` | Look up / down |
| `Space` | Hold the space action |
| `1`–`9` or an event chip | Hold an event: adds `The event is: ...` to the prompt until released |
| Yaw / Pitch speed (Advanced) | Turn and look speed, in degrees per latent frame |
| Prompt + **Send** | Replace the base prompt for upcoming video, keeping the current world; held events build on it |
| Perspective | First person (FPS) or third person (TPS) |
| Seed + **Reset world** (Advanced) | Restart the current image with the chosen seed |
| **Pause** / **Resume** | Pause or resume the video stream |

Controls are held while pressed and released when you let go; the on-screen keys work the same way. Prompt changes and events apply from the next chunk the model has not yet generated, so they show after a short delay; hold an event until you see it. **✎ edit** next to the chips changes the current example's events (saved in this browser). When the chunk budget is used up, reset or pick an example to continue.

## Adding your own example

Click **Add your own example** (or drop an image onto the example list), then choose a PNG, JPEG or WebP image, a name, the perspective, the scene (`The scene in the video is: ...`), for TPS the character (`The character is: ...`), and optional named events (a short description of what happens). Write the scene (and the character for TPS) for every example: the prompt is what the model generates from.

Added examples are saved in this browser's IndexedDB, so they stay across reloads but not across browsers or machines. Delete one with its **×** button. To ship an example with the app, put the image in `public/examples/` and add an entry to `lib/examples.ts`.
