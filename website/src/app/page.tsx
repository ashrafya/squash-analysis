import Navbar from "@/components/layout/Navbar";
import Footer from "@/components/layout/Footer";
import Hero from "@/components/sections/Hero";
import StatsBar from "@/components/sections/StatsBar";
import HowItWorks from "@/components/sections/HowItWorks";
import FeaturesGrid from "@/components/sections/FeaturesGrid";
import SampleOutputs from "@/components/sections/SampleOutputs";
import CTABanner from "@/components/sections/CTABanner";

export default function LandingPage() {
  return (
    <div className="bg-bg-light min-h-screen">
      <Navbar />
      <main>
        <Hero />
        <StatsBar />
        <HowItWorks />
        <FeaturesGrid />
        <SampleOutputs />
        <CTABanner />
      </main>
      <Footer />
    </div>
  );
}
