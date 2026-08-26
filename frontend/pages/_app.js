// import "@/styles/globals.css";

// export default function App({ Component, pageProps }) {
//   return <Component {...pageProps} />;
// }

// Next.js application shell: loads global styles and wraps every page component.
import { useEffect } from 'react'
import '../styles/globals.css'

export default function MyApp({ Component, pageProps }) {
  // Browser-only side effect keeps the tab title independent of individual page content.
  useEffect(() => { document.title = 'DocuMind'}, [])
  return <Component {...pageProps} />
}
