import { Suspense } from 'react'
import LandingPage from '@/components/LandingPage'

export default function HomePage() {
  return (
    <Suspense fallback={null}>
      <LandingPage />
    </Suspense>
  )
}
