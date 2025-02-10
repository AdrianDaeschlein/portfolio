<script lang="ts">
  import { onMount } from 'svelte';
  import '../app.css';

  import SkincancerDetection from '../components/SkincancerDetection.svelte';
  import TableauVisualization from '../components/TableauVisualization.svelte';
  import TableauVisualizationMore from '../components/TableauVisualizationMore.svelte';
  import ReinforcementLearning from '../components/ReinforcementLearning.svelte';
  import PredictiveAnalytics from '../components/PredictiveAnalytics.svelte';
  import PredictiveAnalyticsCode from '../components/PredictiveAnalyticsCode.svelte';
  import SkincancerDetectionCode from '../components/SkincancerDetectionCode.svelte';
  import FaceSwap from '../components/FaceSwap.svelte';
  import FaceSwapCode from '../components/FaceSwapCode.svelte';
  import Isek from '../components/Isek.svelte';
  import IsekMore from '../components/IsekMore.svelte';
  import FallDetection from '../components/FallDetection.svelte';
  import FallDetectionCode from '../components/FallDetectionCode.svelte';
  let activeSubheading: number = 0;
  let mounted = false;
  let animationTriggered = false;

  let centerTransform = '-100%';
  let rightTransform = '-200%';

  // Subheading specific content
  const subheadingContent: { [key: number]: string } = {
    1: 'This is the content for Subheading 1',
    2: 'This is the content for Subheading 2',
    3: 'This is the content for Subheading 3',
  };

  // Change content based on clicked subheading
  function changeContent(subheading: number) {
    if (activeSubheading !== subheading) {
      animationTriggered = false;
      centerTransform = '-100%';
      rightTransform = '-200%';

      setTimeout(() => {
        activeSubheading = subheading;
        animationTriggered = true;
        centerTransform = '0';
        rightTransform = '0';
      }, 50);
    }
  }

  onMount(() => {
    setTimeout(() => {
      activeSubheading = 1;
      mounted = true;
      animationTriggered = true;
      centerTransform = '0';
      rightTransform = '0';
    }, 100);
  });
</script>

<div class="container">
  <!-- Left column -->
  <div
    class="left"
    style="display: flex; flex-direction: column; justify-content: flex-start;"
  >

    <!-- Profile Picture -->
    <img src="/port_pic.png" alt="Profile Picture" class="profile-pic" />
    <h1 style="margin-block-start: 1em;">PROJECTS</h1>
    <div>
      <div
        class="subheadings"
        role="button"
        tabindex="0"
        on:click={() => changeContent(1)}
        on:keydown={(e) => e.key === 'Enter' && changeContent(1)}
      >
        SKINCANCER DETECTION
      </div>
      <div
        class="subheadings"
        role="button"
        tabindex="0"
        on:click={() => changeContent(2)}
        on:keydown={(e) => e.key === 'Enter' && changeContent(2)}
      >
        TABLEAU VISUALIZATION OF AIRBNB DATA
      </div>
      <div
        class="subheadings"
        role="button"
        tabindex="0"
        on:click={() => changeContent(3)}
        on:keydown={(e) => e.key === 'Enter' && changeContent(3)}
      >
        Bachelor: RL MODEL FOR SAME-DAY DELIVERY
      </div>
      <div
        class="subheadings"
        role="button"
        tabindex="0"
        on:click={() => changeContent(4)}
        on:keydown={(e) => e.key === 'Enter' && changeContent(4)}
      >
        PREDICTIVE ANALYTICS OF AIRLINE DELAYS
      </div>
      <div
        class="subheadings"
        role="button"
        tabindex="0"
        on:click={() => changeContent(5)}
        on:keydown={(e) => e.key === 'Enter' && changeContent(4)}
      >
        ISEK - AI CALENDAR
      </div>
      <div
        class="subheadings"
        role="button"
        tabindex="0"
        on:click={() => changeContent(6)}
        on:keydown={(e) => e.key === 'Enter' && changeContent(6)}
      >
        ML for Face Swaps
      </div>
      <div
        class="subheadings"
        role="button"
        tabindex="0"
        on:click={() => changeContent(7)}
        on:keydown={(e) => e.key === 'Enter' && changeContent(7)}
      >
        Master: Fall Detection
      </div>
    </div>
    <br />
    <br />
    <h1>ABOUT ME</h1>
    <div>
      <div class="subheadings">CV</div>
    </div>
  </div>

  <!-- Center column -->
  <div
    class="center"
    class:active={mounted && activeSubheading !== 0}
    class:animate={animationTriggered}
    style="transform: translateX({centerTransform})"
  >
    <div class="center-content">
      {#if activeSubheading === 1}
        <SkincancerDetection />
      {/if}
      {#if activeSubheading === 2}
        <TableauVisualization />
      {/if}
      {#if activeSubheading === 3}
        <ReinforcementLearning />
      {/if}
      {#if activeSubheading === 4}
        <PredictiveAnalytics />
      {/if}
      {#if activeSubheading === 5}
        <Isek />
      {/if}
      {#if activeSubheading === 6}
        <FaceSwap />
      {/if}
      {#if activeSubheading === 7}
        <FallDetection />
      {/if}
    </div>
  </div>

  <!-- Right column -->
  <div
    class="right"
    class:active={mounted && activeSubheading !== 0}
    class:animate={animationTriggered}
    style="transform: translateX({rightTransform})"
  >
    {#if activeSubheading === 1}
      <SkincancerDetectionCode />
    {/if}
    {#if activeSubheading === 2}
      <TableauVisualizationMore />
    {/if}
    {#if activeSubheading === 3}
      <h1>REINFORCEMENT LEARNING MODEL FOR SAME-DAY DELIVERY</h1>
    {/if}
    {#if activeSubheading === 4}
      <PredictiveAnalyticsCode />
    {/if}
    {#if activeSubheading === 5}
      <IsekMore />
    {/if}
    {#if activeSubheading === 6}
      <FaceSwapCode />
    {/if}
    {#if activeSubheading === 7}
      <FallDetectionCode />
    {/if}
  </div>
</div>

<style>
  /**
    black: 212A31
    grey: 2E3944
    blue1: 124E66
    blue2: 748D92
    background: D3D9D4
    */
  .container {
    margin: 0;
    padding: 0;
    background-color: #d3d9d4;
  }

  .container {
    position: relative;
    display: flex;
    width: 100vw;
    height: 100vh;
    overflow: hidden;
  }

  .left {
    position: relative;
    z-index: 3;
    background-color: #d3d9d4;
    width: 20%;
    display: flex;
    flex-direction: column;
    justify-content: center;
    padding-left: 20px;
    padding-top: 40px;
  }

  .center,
  .right {
    position: absolute;
    transition: transform 0s;
    height: 100%;
  }

  .center.animate,
  .right.animate {
    transition: transform 1s ease-in-out;
  }

  .center {
    z-index: 2;
    background-color: #748d92;
    width: 40%;
    height: 100%;
    display: flex;
    flex-direction: column;
    color: white;
    font-size: 24px;
    transform: translateX(-100%);
    left: 20%;
    padding: 40px;
    overflow-y: auto;
    scrollbar-width: none; /* Firefox */
    -ms-overflow-style: none; /* Internet Explorer 10+ */
  }

  .center::-webkit-scrollbar {
    width: 0;
    height: 0;
    display: none; /* Chrome, Safari, and Opera */
  }

  .center-content {
    flex-grow: 1;
    overflow-y: auto;
    scrollbar-width: none; /* Firefox */
    -ms-overflow-style: none; /* Internet Explorer 10+ */
  }

  .center-content::-webkit-scrollbar {
    width: 0;
    height: 0;
    display: none; /* Chrome, Safari, and Opera */
  }

  .right {
    z-index: 1;
    background-color: #124e66;
    width: 40%;
    transform: translateX(-200%);
    display: flex;
    align-items: center;
    justify-content: center;
    color: white;
    font-size: 24px;
    left: 60%;
    padding: 40px;
  }

  .center.active,
  .right.active {
    transform: translateX(0);
  }

  h1 {
    margin: 0 0 20px;
    font-size: 24px;
  }

  .subheadings {
    cursor: pointer;
    font-size: 18px;
    margin: 10px 0;
  }

  .subheadings:hover {
    text-decoration: underline;
  }

  .profile-pic {
    width: 120px; /* Adjust size as needed */
    height: 120px;
    border-radius: 50%;
    object-fit: cover;
    display: block;
    margin: 0 auto 1em auto;
  }
</style>
