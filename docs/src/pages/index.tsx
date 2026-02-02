import type {ReactNode} from 'react';
import {useEffect} from 'react';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Layout from '@theme/Layout';
import Heading from '@theme/Heading';
import CodeBlock from '@theme/CodeBlock';

import styles from './index.module.css';

function useHomepageNavbar() {
  useEffect(() => {
    document.documentElement.classList.add('homepage');
    document.body.classList.add('homepage');

    const navbar = document.querySelector('.navbar') as HTMLElement;

    const updateNavbar = () => {
      if (!navbar) return;
      const scrolled = window.scrollY > 50;

      if (scrolled) {
        document.documentElement.classList.add('scrolled');
        navbar.style.removeProperty('background');
        navbar.style.removeProperty('background-color');
        navbar.style.removeProperty('border-bottom');
        navbar.style.removeProperty('backdrop-filter');
      } else {
        document.documentElement.classList.remove('scrolled');
        navbar.style.setProperty('background', 'transparent', 'important');
        navbar.style.setProperty('background-color', 'transparent', 'important');
        navbar.style.setProperty('border-bottom', 'none', 'important');
        navbar.style.setProperty('backdrop-filter', 'none', 'important');
      }
    };

    updateNavbar();
    window.addEventListener('scroll', updateNavbar);

    return () => {
      document.documentElement.classList.remove('homepage');
      document.documentElement.classList.remove('scrolled');
      document.body.classList.remove('homepage');
      window.removeEventListener('scroll', updateNavbar);
      if (navbar) {
        navbar.style.removeProperty('background');
        navbar.style.removeProperty('background-color');
        navbar.style.removeProperty('border-bottom');
        navbar.style.removeProperty('backdrop-filter');
      }
    };
  }, []);
}

const exampleCode = `from pure_agents import Agent, tool

@tool
def search(query: str) -> str:
    """Search the web."""
    return f"Results for {query}..."

agent = Agent(tools=[search])
result = await agent.run("Find the weather in Madrid")`;

function HomepageHeader() {
  const {siteConfig} = useDocusaurusContext();
  return (
    <header className={styles.hero}>
      <div className="container">
        <div className={styles.heroInner}>
          <Heading as="h1" className={styles.heroTitle}>
            {siteConfig.title}
          </Heading>
          <p className={styles.heroSubtitle}>Powerful, but simple</p>
          <p className={styles.heroDescription}>
            ~800 lines of code. All the features. No complexity.
          </p>
          <div className={styles.heroButtons}>
            <Link className={styles.primaryButton} to="/docs/">
              Get started
              <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
                <path d="M6 3L11 8L6 13" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
              </svg>
            </Link>
            <Link
              className={styles.secondaryButton}
              href="https://github.com/jmbarrancoml/pureagents">
              <svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor">
                <path d="M12 0C5.37 0 0 5.37 0 12c0 5.31 3.435 9.795 8.205 11.385.6.105.825-.255.825-.57 0-.285-.015-1.23-.015-2.235-3.015.555-3.795-.735-4.035-1.41-.135-.345-.72-1.41-1.23-1.695-.42-.225-1.02-.78-.015-.795.945-.015 1.62.87 1.845 1.23 1.08 1.815 2.805 1.305 3.495.99.105-.78.42-1.305.765-1.605-2.67-.3-5.46-1.335-5.46-5.925 0-1.305.465-2.385 1.23-3.225-.12-.3-.54-1.53.12-3.18 0 0 1.005-.315 3.3 1.23.96-.27 1.98-.405 3-.405s2.04.135 3 .405c2.295-1.56 3.3-1.23 3.3-1.23.66 1.65.24 2.88.12 3.18.765.84 1.23 1.905 1.23 3.225 0 4.605-2.805 5.625-5.475 5.925.435.375.81 1.095.81 2.22 0 1.605-.015 2.895-.015 3.3 0 .315.225.69.825.57A12.02 12.02 0 0024 12c0-6.63-5.37-12-12-12z"/>
              </svg>
              GitHub
            </Link>
          </div>
          <div className={styles.installCommand}>
            <span className={styles.installPrefix}>$</span>
            <code>pip install pureagents</code>
          </div>
        </div>
      </div>
    </header>
  );
}

function CodeExample() {
  return (
    <section className={styles.codeSection}>
      <div className="container">
        <div className={styles.codeGrid}>
          <div className={styles.codeDescription}>
            <Heading as="h2">Built to build on</Heading>
            <p>
              Most agent frameworks are over-engineered. Thousands of lines,
              dozens of abstractions, patterns on patterns.
            </p>
            <p>
              <strong>pureagents</strong> gives you all the features without
              the complexity. Fork it, modify it, make it yours.
            </p>
            <ul className={styles.featureList}>
              <li>Every feature is optional and modular</li>
              <li>One parameter = one feature</li>
              <li>No magic, no hidden behaviour</li>
            </ul>
          </div>
          <div className={styles.codeBlock}>
            <CodeBlock language="python">{exampleCode}</CodeBlock>
          </div>
        </div>
      </div>
    </section>
  );
}

function PhilosophySection() {
  return (
    <section className={styles.philosophy}>
      <div className="container">
        <div className={styles.philosophyInner}>
          <div className={styles.philosophyContent}>
            <Heading as="h2">Many features ≠ complicated</Heading>
            <p>
              We believe powerful software can be simple. Every feature in pureagents
              is <strong>optional</strong>, <strong>modular</strong>, and adds exactly
              one parameter to the API.
            </p>
            <p>
              Read the entire codebase in 30 minutes. Understand how everything works.
              Then make it your own.
            </p>
          </div>
          <div className={styles.principlesList}>
            <div className={styles.principle}>
              <div className={styles.principleNumber}>1</div>
              <div className={styles.principleContent}>
                <h3>Read</h3>
                <p>Understand the entire codebase in 30 minutes</p>
              </div>
            </div>
            <div className={styles.principle}>
              <div className={styles.principleNumber}>2</div>
              <div className={styles.principleContent}>
                <h3>Copy</h3>
                <p>Fork and modify, don't depend on upstream</p>
              </div>
            </div>
            <div className={styles.principle}>
              <div className={styles.principleNumber}>3</div>
              <div className={styles.principleContent}>
                <h3>Build</h3>
                <p>Base for your own implementation, not a black box</p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}

// Icons
const ToolsIcon = () => (
  <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M14.7 6.3a1 1 0 0 0 0 1.4l1.6 1.6a1 1 0 0 0 1.4 0l3.77-3.77a6 6 0 0 1-7.94 7.94l-6.91 6.91a2.12 2.12 0 0 1-3-3l6.91-6.91a6 6 0 0 1 7.94-7.94l-3.76 3.76z"/>
  </svg>
);

const ProvidersIcon = () => (
  <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <rect x="2" y="2" width="6" height="6" rx="1"/>
    <rect x="16" y="2" width="6" height="6" rx="1"/>
    <rect x="9" y="16" width="6" height="6" rx="1"/>
    <path d="M5 8v3a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V8"/>
    <path d="M12 13v3"/>
  </svg>
);

const StreamingIcon = () => (
  <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M4 6h16"/>
    <path d="M4 12h16"/>
    <path d="M4 18h10"/>
    <circle cx="19" cy="18" r="2" fill="currentColor"/>
  </svg>
);

const MemoryIcon = () => (
  <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M4 4h16c1.1 0 2 .9 2 2v12c0 1.1-.9 2-2 2H4c-1.1 0-2-.9-2-2V6c0-1.1.9-2 2-2z"/>
    <path d="M8 10h8"/>
    <path d="M8 14h6"/>
  </svg>
);

const StructuredIcon = () => (
  <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <rect x="3" y="3" width="18" height="18" rx="2"/>
    <path d="M3 9h18"/>
    <path d="M9 21V9"/>
  </svg>
);

const HooksIcon = () => (
  <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <circle cx="12" cy="5" r="3"/>
    <circle cx="5" cy="19" r="3"/>
    <circle cx="19" cy="19" r="3"/>
    <path d="M12 8v4l-4.5 4"/>
    <path d="M12 12l4.5 4"/>
  </svg>
);

const BatchIcon = () => (
  <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <rect x="2" y="4" width="5" height="5" rx="1"/>
    <rect x="9.5" y="4" width="5" height="5" rx="1"/>
    <rect x="17" y="4" width="5" height="5" rx="1"/>
    <path d="M4.5 12v3"/>
    <path d="M12 12v3"/>
    <path d="M19.5 12v3"/>
    <path d="M4.5 18h15"/>
  </svg>
);

const RetryIcon = () => (
  <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M3 12a9 9 0 1 0 9-9 9.75 9.75 0 0 0-6.74 2.74L3 8"/>
    <path d="M3 3v5h5"/>
  </svg>
);

const UsageIcon = () => (
  <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M3 3v18h18"/>
    <path d="M7 16l4-4 4 4 6-6"/>
  </svg>
);

const ChainIcon = () => (
  <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M10 13a5 5 0 0 0 7.54.54l3-3a5 5 0 0 0-7.07-7.07l-1.72 1.71"/>
    <path d="M14 11a5 5 0 0 0-7.54-.54l-3 3a5 5 0 0 0 7.07 7.07l1.71-1.71"/>
  </svg>
);

const RouterIcon = () => (
  <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M12 2v4"/>
    <path d="M12 18v4"/>
    <path d="M4.93 4.93l2.83 2.83"/>
    <path d="M16.24 16.24l2.83 2.83"/>
    <path d="M2 12h4"/>
    <path d="M18 12h4"/>
    <path d="M4.93 19.07l2.83-2.83"/>
    <path d="M16.24 7.76l2.83-2.83"/>
    <circle cx="12" cy="12" r="4"/>
  </svg>
);

const PlanIcon = () => (
  <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M9 5H7a2 2 0 0 0-2 2v12a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V7a2 2 0 0 0-2-2h-2"/>
    <rect x="9" y="3" width="6" height="4" rx="1"/>
    <path d="M9 12l2 2 4-4"/>
  </svg>
);

const GraphIcon = () => (
  <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <circle cx="5" cy="6" r="3"/>
    <circle cx="19" cy="6" r="3"/>
    <circle cx="12" cy="18" r="3"/>
    <path d="M7.5 7.5l3 6"/>
    <path d="M16.5 7.5l-3 6"/>
  </svg>
);

type FeatureItem = {
  title: string;
  description: string;
  Icon: () => ReactNode;
};

const FeatureList: FeatureItem[] = [
  {
    title: 'Tools',
    description: '@tool decorator. Type hints become JSON schemas automatically.',
    Icon: ToolsIcon,
  },
  {
    title: 'Providers',
    description: 'Mistral, OpenAI, Anthropic. Switch with one parameter.',
    Icon: ProvidersIcon,
  },
  {
    title: 'Streaming',
    description: 'Real-time responses, token by token.',
    Icon: StreamingIcon,
  },
  {
    title: 'Memory',
    description: 'Persist conversations. Built-in or bring your own.',
    Icon: MemoryIcon,
  },
  {
    title: 'Structured outputs',
    description: 'Get typed responses with dataclasses.',
    Icon: StructuredIcon,
  },
  {
    title: 'Hooks',
    description: 'Monitor tool calls, results, and reasoning.',
    Icon: HooksIcon,
  },
  {
    title: 'Batch',
    description: 'Run multiple prompts in parallel.',
    Icon: BatchIcon,
  },
  {
    title: 'Chaining',
    description: 'Run agents in sequence. Output feeds into the next.',
    Icon: ChainIcon,
  },
  {
    title: 'Routing',
    description: 'Direct prompts to specialised agents.',
    Icon: RouterIcon,
  },
  {
    title: 'Planning',
    description: 'Create a plan before executing.',
    Icon: PlanIcon,
  },
  {
    title: 'Graph',
    description: 'Multi-agent workflows with conditional routing.',
    Icon: GraphIcon,
  },
  {
    title: 'Reliability',
    description: 'Retry with backoff, timeouts, context limits.',
    Icon: RetryIcon,
  },
  {
    title: 'Usage tracking',
    description: 'Token counts and cost estimates.',
    Icon: UsageIcon,
  },
];

function Feature({title, description, Icon}: FeatureItem) {
  return (
    <div className={styles.featureCard}>
      <div className={styles.featureIcon}>
        <Icon />
      </div>
      <Heading as="h3" className={styles.featureTitle}>{title}</Heading>
      <p className={styles.featureDescription}>{description}</p>
    </div>
  );
}

function HomepageFeatures() {
  return (
    <section className={styles.features}>
      <div className="container">
        <div className={styles.featuresHeader}>
          <Heading as="h2">Everything you need</Heading>
          <p>All the features for production. Each one optional and modular.</p>
        </div>
        <div className={styles.featuresGrid}>
          {FeatureList.map((props, idx) => (
            <Feature key={idx} {...props} />
          ))}
        </div>
      </div>
    </section>
  );
}

function CTASection() {
  return (
    <section className={styles.cta}>
      <div className="container">
        <div className={styles.ctaInner}>
          <Heading as="h2">Start building</Heading>
          <p>Get up and running in minutes. Read the docs, explore the examples.</p>
          <div className={styles.heroButtons}>
            <Link className={styles.primaryButton} to="/docs/quickstart">
              Quick start
              <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
                <path d="M6 3L11 8L6 13" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
              </svg>
            </Link>
            <Link className={styles.secondaryButton} to="/docs/api/agent">
              API reference
            </Link>
          </div>
        </div>
      </div>
    </section>
  );
}

export default function Home(): ReactNode {
  useHomepageNavbar();

  return (
    <Layout
      title="Powerful, but simple"
      description="pureagents: LLM agent framework. Many features, ~1,500 lines. Built to build on.">
      <HomepageHeader />
      <main>
        <CodeExample />
        <HomepageFeatures />
        <PhilosophySection />
        <CTASection />
      </main>
    </Layout>
  );
}
