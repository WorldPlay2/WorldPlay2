// Server component, read on each request: the endpoint picker (PageClient), built
// from endpoints.json plus the untracked endpoints.local.json; editing either file
// takes effect on reload.
import PageClient from "./PageClient";
import { publicEndpoints } from "@/lib/endpoints-config";

export const dynamic = "force-dynamic";

export default function WorldPlay2Page() {
  return <PageClient endpoints={publicEndpoints()} />;
}
