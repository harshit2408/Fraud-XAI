const { spawn } = require("child_process");
const proc = spawn("ecc-memory-mcp.cmd", [], {
  env: { ...process.env, ECC_MEMORY_HARNESS: "claude" },
  stdio: ["pipe", "pipe", "pipe"],
  shell: true
});

proc.stdout.on("data", d => console.log("STDOUT:", d.toString()));
proc.stderr.on("data", d => console.log("STDERR:", d.toString()));
proc.on("error", e => console.log("SPAWN ERROR:", e));

proc.stdin.write(JSON.stringify({
  jsonrpc: "2.0", id: 1, method: "initialize",
  params: { protocolVersion: "2024-11-05", capabilities: {}, clientInfo: { name: "test", version: "1.0" } }
}) + "\n");

setTimeout(() => {
  // REQUIRED: tell the server initialization is complete
  proc.stdin.write(JSON.stringify({
    jsonrpc: "2.0", method: "notifications/initialized"
  }) + "\n");
}, 500);

setTimeout(() => {
  proc.stdin.write(JSON.stringify({ jsonrpc: "2.0", id: 2, method: "tools/list" }) + "\n");
}, 1500);

setTimeout(() => {
  proc.stdin.write(JSON.stringify({
    jsonrpc: "2.0", id: 3, method: "tools/call",
    params: { name: "memory_search", arguments: { query: "fraud-xai" } }
  }) + "\n");
}, 2500);

setTimeout(() => process.exit(0), 4500);
