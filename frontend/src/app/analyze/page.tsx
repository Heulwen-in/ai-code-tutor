import { AppShell } from "@/components/AppShell";
import { TutorWorkspace } from "@/components/TutorWorkspace";

export default function AnalyzePage() {
  return (
    <AppShell
      title="AI Program Tutor"
      description="Write Python code on the left. Review bug feedback and lesson recommendations on the right."
    >
      <TutorWorkspace />
    </AppShell>
  );
}
