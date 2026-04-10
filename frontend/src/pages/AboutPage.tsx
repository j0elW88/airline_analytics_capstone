/**
 * @file src/pages/AboutPage.tsx
 * @description Simple "About" page with project and developer bios.
 */

import { PageShell } from "../components/layout/PageShell";
import { Card } from "../components/ui/Card";

const bios = [
  {
    name: "Joel Woeste",
    role: "Team Lead & Backend",
    bio: "Leads coordination and backend analytics logic. Handled market-power computation integration and output structure alignment.",
  },
  {
    name: "Gavin Schroeder",
    role: "Scrum Master & Backend",
    bio: "Responsible for backend ingestion and aggregation. Implemented chunked processing and output generation for route and hub datasets.",
  },
  {
    name: "Nitin Guhan",
    role: "Frontend & Backend Developer",
    bio: "Handled backend/frontend integration and analytics flow. Completed app-connected load and single-period analysis workflows.",
  },
  {
    name: "Lauren Nunag",
    role: "Frontend Developer",
    bio: "Designed UI and results presentation. Designed results layout, filter interactions, and analytics display components.",
  },
  {
    name: "Lynn Chen",
    role: "Frontend Developer",
    bio: "Focused on QA-facing validation and project documentation. Performed issue tracking and behavior verification for key flows.",
  },
];

export function AboutPage() {
  return (
    <PageShell
      title="About"
      subtitle="Airline Analytics Capstone (Blitz Analytics)"
    >
      <section style={{ display: "grid", gap: 12 }}>
        <Card>
          <h3 style={{ marginTop: 0 }}>Project</h3>
          <p style={{ marginBottom: 0 }}>
            This dashboard supports exploring airline market concentration and pricing signals using DB1B Market data.
            It combines a Python parsing/analysis pipeline with a modular React UI.
          </p>
        </Card>

        <Card>
          <h3 style={{ marginTop: 0 }}>Developers</h3>
          <div style={{ display: "grid", gap: 10 }}>
            {bios.map((dev) => (
              <div key={dev.name} style={{ borderTop: "1px solid var(--blank)", paddingTop: 10 }}>
                <div style={{ display: "flex", justifyContent: "space-between", gap: 12, flexWrap: "wrap" }}>
                  <strong>{dev.name}</strong>
                  <span style={{ color: "var(--muted-foreground)", fontWeight: 600 }}>{dev.role}</span>
                </div>
                <p style={{ margin: "6px 0 0" }}>{dev.bio}</p>
              </div>
            ))}
          </div>
        </Card>
      </section>
    </PageShell>
  );
}

