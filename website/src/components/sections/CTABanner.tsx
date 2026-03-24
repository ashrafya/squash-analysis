import Link from "next/link";
import Button from "@/components/ui/Button";
import { ArrowRight } from "lucide-react";

export default function CTABanner() {
  return (
    <section className="bg-lime border-b-2 border-ink">
      <div className="max-w-6xl mx-auto px-4 sm:px-6 py-20">
        <div className="flex flex-col md:flex-row items-start md:items-end justify-between gap-10">
          <div>
            <h2 className="font-heading font-extrabold text-5xl sm:text-6xl text-ink leading-tight">
              Ready to analyse<br/>
              <span className="text-coral">your match?</span>
            </h2>
            <p className="mt-4 text-ink/60 font-body max-w-md">
              No account needed. No video leaves your machine. Upload an MP4
              and get a full report in minutes.
            </p>
          </div>
          <div className="flex-shrink-0">
            <Link href="/analyze">
              <Button size="lg" variant="dark">
                Upload a Match <ArrowRight className="w-5 h-5"/>
              </Button>
            </Link>
          </div>
        </div>
      </div>
    </section>
  );
}
