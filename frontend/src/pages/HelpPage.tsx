/**
 * @file src/pages/HelpPage.tsx
 */
import { PageShell } from "../components/layout/PageShell";
import { AppButton } from "../components/ui/AppButton";

interface HelpPageProps {
  onBack: () => void;
}

export function HelpPage({ onBack }: HelpPageProps) {
  return (
    <PageShell 
      title="Help Guide" 
      subtitle="User instructions for Airline Analytics Capstone"
      actions={
        <AppButton variant="ghost" onClick={onBack}>
          Back
        </AppButton>
      }
    >
      <section className="help-content" style={{ maxWidth: '1200px', margin: '0 auto', padding: '0 20px' }}>
        <div style={{ 
          display: 'flex', 
          flexDirection: 'row', 
          gap: '2.5rem', 
          alignItems: 'flex-start',
          textAlign: 'left' 
        }}>

        <div className="help-section" style={{ flex: 1 }}>
            <h3> (1) BTS Data Retrieval</h3>
            <p>Navigate to the 
              <a href="https://www.transtats.bts.gov/databases.asp?Z1qr_VQ=E&Z1qr_Qr5p=N8vn6v10&f7owrp6_VQF=D" target="_blank" rel="noopener noreferrer"> Bureau of Transportation Statistics website </a> 
              and find the "Airline Origin and Destination Survey (DB1BMarket)" table. </p> <p> Select preferred <code>Year</code> and <code>Quarter</code> you wish to investigate. Then download <strong>DB1B Market</strong> data. </p> <p> Then ensure the following columns are selected:</p>
            <ul style={{ paddingLeft: '1.2rem' }}>
              <li><code>Year</code></li>
              <li><code>Quarter</code></li>
              <li><code>Origin</code></li>
              <li><code>OriginState</code></li>     
              <li><code>Dest</code></li>                    
              <li><code>DestState</code></li>
              <li><code>TkCarrier</code></li>            
              <li><code>Passengers</code></li>
              <li><code>MktFare</code></li> 
              <li><code> NonStopMiles </code></li>
            </ul>
          </div>

          <div className="help-section" style={{ flex: 1 }}>
            <h3> (2) Load Data</h3>
            <p>Click on <strong>Load Data Set</strong>. The "entry point" for analysis. Use this to move raw BTS data into the system. </p>
            <p> <strong>Two methods:</strong> <br />
            <ul>
            <li>Raw Upload: Select a DB1B Market CSV. The backend validates columns like <code>MktFare</code> and <code>Passengers</code>.</li>
            <li>Existing Periods: If a dataset was previously parsed, select it from the dropdown to skip the upload phase.</li>
            </ul> 
            </p>  
          </div>

          <div className="help-section" style={{ flex: 1 }}>
            <h3> (3) Analyze Periods</h3>
            <p>Click on either <strong>Analyze One Period</strong> or <strong>Analyze Multiple Periods</strong> to start gaining key insights once data is loaded.</p>
            <p><strong>Analyze One:</strong> Results in a snapshot of a single quarter. It calculates the HHI (Market Concentration) and weighted average fares for specific routes.</p>
            <p><strong>Analyze Multiple:</strong> Results in a time-series view. Compares multiple quarters to track fare volatility and carrier market share shifts over time.</p>
          </div>

        </div>

      </section>
    </PageShell>
  );
}