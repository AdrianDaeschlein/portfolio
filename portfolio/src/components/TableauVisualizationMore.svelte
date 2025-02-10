<script lang="ts">
  import { onMount } from 'svelte';

  // Prism.js for syntax highlighting
  onMount(async () => {});
</script>

<div class="code-container">
  <div class="header">
    <h3>Dashboard</h3>
    <a
      href="https://public.tableau.com/app/profile/adrian6377/viz/Final_17044137154000/MainStory?publish=yes"
      target="_blank"
      rel="noopener noreferrer"
      title="Open in Tableau Public"
    >
      <img
        src="/tableau-clipart.png"
        alt="Tableau Public"
        class="tableau-icon"
      />
    </a>
  </div>
  <div class="dashboard-images">
    <div class="dashboard-image">
      <img src="/dashboard1.png" alt="Dashboard 1" class="macbook-screenshot" />
      <p class="image-caption">Dashboard 1: Overview</p>
    </div>
    <div class="dashboard-image">
      <img src="/dashboard2.png" alt="Dashboard 2" class="macbook-screenshot" />
      <p class="image-caption">Dashboard 2: Filtering</p>
    </div>
  </div>
</div>

<style>
  .code-container {
    display: flex;
    flex-direction: column;
    align-items: flex-start;
    width: 100%;
    height: 100%;
    margin: 40px;
    margin-left: 80px;
  }

  .header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    width: 100%;
    max-width: 400px;
  }

  .tableau-icon {
    width: 48px;
    height: 24px;
    margin-left: 10px;
  }

  .dashboard-images {
    display: flex;
    flex-direction: column;
    align-items: center;
    margin-top: 20px;
  }

  .dashboard-image {
    margin-bottom: 20px;
    margin-right: 20px;
  }

  .macbook-screenshot {
    width: 100%;
    height: auto;
    border-radius: 10px;
  }

  .image-caption {
    margin-top: 10px;
    font-style: italic;
  }
</style>
