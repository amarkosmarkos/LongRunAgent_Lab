// Visual identity for each agent role, shared by the graph, story feed,
// cost bars and detail panels so "who did what" reads at a glance.
export const AGENTS = {
  planner: {
    label: "Planner", color: "var(--agent-planner)", initials: "PL",
    blurb: "defines scope and success criteria",
  },
  strategist: {
    label: "Strategist", color: "var(--agent-strategist)", initials: "ST",
    blurb: "proposes the hypotheses that become branches",
  },
  experimenter: {
    label: "Experimenter", color: "var(--agent-experimenter)", initials: "EX",
    blurb: "writes and runs solver code",
  },
  critic: {
    label: "Critic", color: "var(--agent-critic)", initials: "CR",
    blurb: "judges results and extracts shared insights",
  },
  supervisor: {
    label: "Supervisor", color: "var(--agent-supervisor)", initials: "SU",
    blurb: "collapses, merges and declares winners",
  },
  researcher: {
    label: "Researcher", color: "var(--agent-researcher)", initials: "RE",
    blurb: "searches the web for state-of-the-art approaches",
  },
};

export function agentMeta(name) {
  if (!name) return { label: "Engine", color: "var(--faint)", initials: "::" };
  return AGENTS[name] || {
    label: name, color: "var(--faint)", initials: name.slice(0, 2).toUpperCase(),
  };
}
