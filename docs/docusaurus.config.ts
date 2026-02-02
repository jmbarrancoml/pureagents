import {themes as prismThemes} from 'prism-react-renderer';
import type {Config} from '@docusaurus/types';
import type * as Preset from '@docusaurus/preset-classic';

const config: Config = {
  title: 'pureagents',
  tagline: 'LLM agents without the complexity',
  favicon: 'img/favicon.ico',

  future: {
    v4: true,
  },

  url: 'https://pureagents.dev',
  baseUrl: '/',

  organizationName: 'jmbarrancoml',
  projectName: 'pureagents',

  onBrokenLinks: 'throw',

  i18n: {
    defaultLocale: 'en',
    locales: ['en'],
  },

  presets: [
    [
      'classic',
      {
        docs: {
          sidebarPath: './sidebars.ts',
          editUrl: 'https://github.com/jmbarrancoml/pureagents/tree/main/docs/',
        },
        blog: false,
        theme: {
          customCss: './src/css/custom.css',
        },
      } satisfies Preset.Options,
    ],
  ],

  themeConfig: {
    image: 'img/social-card.png',
    colorMode: {
      defaultMode: 'dark',
      respectPrefersColorScheme: true,
    },
    metadata: [
      {name: 'keywords', content: 'llm, agents, python, ai, framework, tools, mistral, openai, anthropic'},
      {name: 'twitter:card', content: 'summary_large_image'},
    ],
    navbar: {
      title: 'pureagents',
      logo: {
        alt: 'pureagents',
        src: 'img/logo.svg',
      },
      items: [
        {
          type: 'docSidebar',
          sidebarId: 'docsSidebar',
          position: 'left',
          label: 'Docs',
        },
        {
          href: 'https://github.com/jmbarrancoml/pureagents',
          label: 'GitHub',
          position: 'right',
        },
      ],
    },
    footer: {
      style: 'light',
      links: [
        {
          title: 'Docs',
          items: [
            {
              label: 'Quick start',
              to: '/docs/quickstart',
            },
            {
              label: 'API reference',
              to: '/docs/api/agent',
            },
          ],
        },
        {
          title: 'More',
          items: [
            {
              label: 'GitHub',
              href: 'https://github.com/jmbarrancoml/pureagents',
            },
            {
              label: 'PyPI',
              href: 'https://pypi.org/project/pureagents/',
            },
          ],
        },
      ],
      copyright: `Copyright © ${new Date().getFullYear()} Jose Manuel Flores Barranco. Apache 2.0.`,
    },
    prism: {
      theme: prismThemes.github,
      darkTheme: prismThemes.oneDark,
      additionalLanguages: ['python', 'bash', 'json'],
      magicComments: [
        {
          className: 'code-block-highlighted-line',
          line: 'highlight-next-line',
          block: {start: 'highlight-start', end: 'highlight-end'},
        },
      ],
    },
  } satisfies Preset.ThemeConfig,
};

export default config;
