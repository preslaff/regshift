<script lang="ts">
  import { Router, Route } from 'svelte-routing'
  import { onMount } from 'svelte'
  
  // Layout components
  import Sidebar from './lib/components/layout/Sidebar.svelte'
  import Header from './lib/components/layout/Header.svelte'
  import LoadingSpinner from './lib/components/common/LoadingSpinner.svelte'
  
  // Page components
  import Dashboard from './lib/pages/Dashboard.svelte'
  import Portfolio from './lib/pages/Portfolio.svelte'
  import Regimes from './lib/pages/Regimes.svelte'
  import Backtesting from './lib/pages/Backtesting.svelte'
  import Scenarios from './lib/pages/Scenarios.svelte'
  import Analytics from './lib/pages/Analytics.svelte'
  import Settings from './lib/pages/Settings.svelte'
  import Login from './lib/pages/Login.svelte'
  
  // Stores and services
  import { authStore } from './lib/stores/auth'
  import { themeStore } from './lib/stores/theme'
  import { apiService } from './lib/services/api'
  
  let isLoading = true
  let sidebarCollapsed = false
  
  onMount(async () => {
    // Initialize theme
    themeStore.initialize()
    
    // Check authentication status
    await authStore.initialize()
    
    isLoading = false
  })
  
  function toggleSidebar() {
    sidebarCollapsed = !sidebarCollapsed
  }
  
  $: isAuthenticated = $authStore.isAuthenticated
  $: currentTheme = $themeStore.theme
</script>

<div class="app" class:dark={currentTheme === 'dark'}>
  {#if isLoading}
    <div class="loading-screen">
      <LoadingSpinner size="large" />
      <p class="mt-4 text-gray-600">Loading Dynamic Investment Strategies...</p>
    </div>
  {:else}
    <Router>
      {#if !isAuthenticated}
        <Route path="*" component={Login} />
      {:else}
        <div class="app-layout">
          <!-- Sidebar -->
          <div class="sidebar-container" class:collapsed={sidebarCollapsed}>
            <Sidebar {sidebarCollapsed} />
          </div>
          
          <!-- Main content area -->
          <div class="main-container" class:expanded={sidebarCollapsed}>
            <!-- Header -->
            <Header {toggleSidebar} {sidebarCollapsed} />
            
            <!-- Page content -->
            <main class="page-content">
              <Route path="/" component={Dashboard} />
              <Route path="/dashboard" component={Dashboard} />
              <Route path="/portfolio" component={Portfolio} />
              <Route path="/regimes" component={Regimes} />
              <Route path="/backtesting" component={Backtesting} />
              <Route path="/scenarios" component={Scenarios} />
              <Route path="/analytics" component={Analytics} />
              <Route path="/settings" component={Settings} />
            </main>
          </div>
        </div>
      {/if}
    </Router>
  {/if}
</div>

<style>
  .app {
    min-height: 100vh;
    background-color: #f9fafb;
  }
  
  .loading-screen {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    min-height: 100vh;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
  }
  
  .app-layout {
    display: flex;
    min-height: 100vh;
  }
  
  .sidebar-container {
    width: 280px;
    transition: width 0.3s ease;
    background: white;
    border-right: 1px solid #e5e7eb;
  }
  
  .sidebar-container.collapsed {
    width: 80px;
  }
  
  .main-container {
    flex: 1;
    display: flex;
    flex-direction: column;
    overflow: hidden;
    margin-left: 0;
    transition: margin-left 0.3s ease;
  }
  
  .main-container.expanded {
    margin-left: 0;
  }
  
  .page-content {
    flex: 1;
    padding: 24px;
    overflow-y: auto;
    background-color: #f9fafb;
  }
  
  /* Dark theme styles */
  :global(.dark) {
    background-color: #1f2937;
    color: #f9fafb;
  }
  
  :global(.dark .sidebar-container) {
    background-color: #374151;
    border-right-color: #4b5563;
  }
  
  :global(.dark .page-content) {
    background-color: #1f2937;
  }
  
  :global(.dark .card) {
    background-color: #374151;
    border-color: #4b5563;
    color: #f9fafb;
  }
  
  /* Responsive design */
  @media (max-width: 1024px) {
    .sidebar-container {
      width: 280px;
      position: fixed;
      top: 0;
      left: 0;
      height: 100vh;
      z-index: 50;
      transform: translateX(-100%);
      transition: transform 0.3s ease;
    }
    
    .sidebar-container:not(.collapsed) {
      transform: translateX(0);
    }
    
    .main-container {
      margin-left: 0;
      width: 100%;
    }
  }
  
  @media (max-width: 768px) {
    .page-content {
      padding: 16px;
    }
  }
</style>