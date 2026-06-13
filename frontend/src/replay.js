// Pure reducer: events[0..cursor] -> view state. Replay = move the cursor.

export function emptyState() {
  return {
    config: null,
    problem: null,
    instance: null,
    baseline: null,
    scope: null,
    hypotheses: [],
    branches: {}, // id -> {…public, lane, parent_ids, status, best_score, …}
    branchOrder: [],
    insights: [],
    critiques: [],
    decisions: [],
    costs: { total: 0, byAgent: {}, byBranch: {}, calls: 0 },
    results: null,
    winnerBranchId: null,
    bestScore: null,
    bestBranchId: null,
    bestSolution: null,
    endedStatus: null,
    activity: {}, // key -> {agent, action, round, branch_id} of agents thinking now
    models: [], // distinct LLM models seen
    mockMode: null, // null=unknown, true=mock LLM, false=real Claude
  };
}

// one agent works per branch at a time, so the branch id is a stable key;
// run-level agents (planner/strategist/supervisor) key on their own name
const actKey = (agent, branchId) => branchId || "@" + agent;

export function reduceEvents(events, cursor) {
  const s = emptyState();
  const upto = cursor == null ? events.length : Math.min(cursor, events.length);
  for (let i = 0; i < upto; i++) apply(s, events[i]);
  return s;
}

function apply(s, ev) {
  const p = ev.payload || {};
  switch (ev.type) {
    case "run.created":
      s.config = p.config;
      s.problem = p.problem;
      s.instance = p.instance;
      s.baseline = p.baseline;
      break;
    case "agent.thinking":
      s.activity[actKey(ev.agent, ev.branch_id)] = {
        agent: ev.agent, branch_id: ev.branch_id,
        action: p.action, round: p.round, seq: ev.seq,
      };
      break;
    case "scope.defined":
      s.scope = p.scope;
      delete s.activity["@planner"];
      break;
    case "hypotheses.proposed":
      s.hypotheses = p.hypotheses || [];
      delete s.activity["@strategist"];
      break;
    case "experiment.started": {
      // experimenter has moved from authoring code to running it in the sandbox
      const a = s.activity[ev.branch_id];
      if (a) a.action = "running the solver in the sandbox";
      break;
    }
    case "branch.created": {
      const b = p.branch;
      s.branches[b.id] = {
        ...b,
        lane: s.branchOrder.length,
        createdSeq: ev.seq,
        risk: p.risk,
        mergeReason: p.merge_reason,
      };
      s.branchOrder.push(b.id);
      break;
    }
    case "experiment.completed": {
      delete s.activity[ev.branch_id];
      const b = s.branches[ev.branch_id];
      if (b) {
        b.experiments = (b.experiments || 0) + 1;
        if (p.valid && p.improved) {
          b.best_score = p.score;
          if (s.bestScore == null || p.score < s.bestScore) {
            s.bestScore = p.score;
            s.bestBranchId = ev.branch_id;
            if (p.solution) s.bestSolution = p.solution;
          }
        }
      }
      break;
    }
    case "critique.added":
      delete s.activity[ev.branch_id];
      s.critiques.push({ ...p, branch_id: ev.branch_id, seq: ev.seq });
      break;
    case "insight.added":
      s.insights.push(p.insight);
      break;
    case "supervisor.decision":
      delete s.activity["@supervisor"];
      s.decisions.push({ ...p, seq: ev.seq });
      break;
    case "branch.collapsed": {
      const b = s.branches[ev.branch_id];
      if (b) {
        b.status = "collapsed";
        b.collapseReason = p.reason;
      }
      break;
    }
    case "branch.merged":
      (p.source_ids || []).forEach((id) => {
        if (s.branches[id]) s.branches[id].status = "merged";
      });
      break;
    case "branch.winner": {
      const b = s.branches[ev.branch_id];
      if (b) b.status = "winner";
      s.winnerBranchId = ev.branch_id;
      break;
    }
    case "llm.called":
      s.costs.total = p.total_cost_usd ?? s.costs.total + (p.cost_usd || 0);
      s.costs.calls += 1;
      if (ev.agent)
        s.costs.byAgent[ev.agent] =
          (s.costs.byAgent[ev.agent] || 0) + (p.cost_usd || 0);
      if (ev.branch_id)
        s.costs.byBranch[ev.branch_id] =
          (s.costs.byBranch[ev.branch_id] || 0) + (p.cost_usd || 0);
      if (p.model) {
        if (!s.models.includes(p.model)) s.models.push(p.model);
        if (p.model === "mock") { if (s.mockMode == null) s.mockMode = true; }
        else s.mockMode = false; // any real model means this run hit the API
      }
      break;
    case "run.completed":
      s.results = p.results;
      s.endedStatus = "completed";
      s.activity = {}; // nothing is thinking once the run has ended
      break;
    case "run.failed":
      s.endedStatus = "failed";
      s.failure = p.error;
      s.activity = {};
      break;
    default:
      break;
  }
}

// ---- graph model: deterministic git-style lane layout -------------------
// Returns { nodes, edges, lanes } where each node has {x,y} grid coords.
export function buildGraph(events, cursor) {
  const upto = cursor == null ? events.length : Math.min(cursor, events.length);
  const nodes = [];
  const edges = [];
  const lanes = {}; // branch_id -> lane index
  const lastNode = {}; // branch_id -> node id
  const branchMeta = {};
  let row = 0;
  let rootId = null;

  const push = (node) => {
    nodes.push(node);
    return node.id;
  };

  for (let i = 0; i < upto; i++) {
    const ev = events[i];
    const p = ev.payload || {};
    switch (ev.type) {
      case "scope.defined": {
        rootId = push({
          id: `n${ev.seq}`, kind: "scope", lane: -1, row: row++,
          label: "Scope", seq: ev.seq,
        });
        break;
      }
      case "branch.created": {
        const b = p.branch;
        lanes[b.id] = Object.keys(lanes).length;
        branchMeta[b.id] = { name: b.name, strategy: b.strategy };
        const id = push({
          id: `n${ev.seq}`, kind: "created", lane: lanes[b.id], row: row++,
          label: b.name, branch_id: b.id, seq: ev.seq,
        });
        if (b.parent_ids && b.parent_ids.length) {
          b.parent_ids.forEach((pid) => {
            if (lastNode[pid]) edges.push({ from: lastNode[pid], to: id, kind: "merge" });
          });
        } else if (rootId) {
          edges.push({ from: rootId, to: id, kind: "spawn" });
        }
        lastNode[b.id] = id;
        break;
      }
      case "experiment.retry": {
        // a recoverable failure that was re-asked immediately — draw the loop
        const id = push({
          id: `n${ev.seq}`, kind: "retry", lane: lanes[ev.branch_id] ?? 0,
          row: row++, label: p.reason || "retry", branch_id: ev.branch_id,
          seq: ev.seq, retry: p.retry, maxAttempts: p.max_attempts, round: p.round,
        });
        if (lastNode[ev.branch_id])
          edges.push({ from: lastNode[ev.branch_id], to: id, kind: "retry" });
        lastNode[ev.branch_id] = id;
        break;
      }
      case "experiment.completed": {
        const kind = !p.valid ? "failed" : p.improved ? "improved" : "neutral";
        const id = push({
          id: `n${ev.seq}`, kind, lane: lanes[ev.branch_id] ?? 0, row: row++,
          label: p.valid ? Number(p.score).toFixed(1) : "error",
          branch_id: ev.branch_id, seq: ev.seq,
          score: p.score, round: p.round, beats: p.beats_baseline,
        });
        if (lastNode[ev.branch_id])
          edges.push({ from: lastNode[ev.branch_id], to: id,
            kind: p.retries ? "retry" : "line" });
        lastNode[ev.branch_id] = id;
        break;
      }
      case "branch.collapsed": {
        const id = push({
          id: `n${ev.seq}`, kind: "collapsed", lane: lanes[ev.branch_id] ?? 0,
          row: row++, label: "collapsed", branch_id: ev.branch_id, seq: ev.seq,
        });
        if (lastNode[ev.branch_id])
          edges.push({ from: lastNode[ev.branch_id], to: id, kind: "line" });
        lastNode[ev.branch_id] = id;
        break;
      }
      case "branch.winner": {
        const id = push({
          id: `n${ev.seq}`, kind: "winner", lane: lanes[ev.branch_id] ?? 0,
          row: row++, label: "winner", branch_id: ev.branch_id, seq: ev.seq,
        });
        if (lastNode[ev.branch_id])
          edges.push({ from: lastNode[ev.branch_id], to: id, kind: "line" });
        lastNode[ev.branch_id] = id;
        break;
      }
      default:
        break;
    }
  }
  return { nodes, edges, lanes, branchMeta, rows: row };
}
