/** Model name on a hosted endpoint. */
export const MODEL_NAME = "worldplay2";

/**
 * One connection target, as listed in `endpoints.json` (plus the optional,
 * untracked `endpoints.local.json`).
 *
 * - `local`: the model's own HTTP runtime (`reactor run`); no API key or token.
 * - `hosted`: a Reactor API; the server route `app/api/token` mints a session
 *   token with the API key held in the env var named by `apiKeyEnv`.
 */
export type EndpointKind = "local" | "hosted";

/** What the browser receives: the server keeps `apiKeyEnv` to itself. */
export type Endpoint = {
  id: string;
  label: string;
  url: string;
  kind: EndpointKind;
  /** `hosted` only: the model's name on that API, when it differs from MODEL_NAME (e.g. a canonical `org/name`). */
  modelName?: string;
};
