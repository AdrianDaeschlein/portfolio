<script lang="ts">
    import { onMount } from 'svelte';
    import '../app.css';

    import SkincancerDetection from '../components/SkincancerDetection.svelte';
    import TableauVisualization from '../components/TableauVisualization.svelte';
    import ReinforcementLearning from '../components/ReinforcementLearning.svelte';
    import PredictiveAnalytics from '../components/PredictiveAnalytics.svelte';

    let activeSubheading: number = 0;
    let mounted = false;
    let animationTriggered = false;

    let centerTransform = '-100%';
    let rightTransform = '-200%';

    // Subheading specific content
    const subheadingContent: { [key: number]: string } = {
        1: "This is the content for Subheading 1",
        2: "This is the content for Subheading 2",
        3: "This is the content for Subheading 3",
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
        background-color: #D3D9D4;
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
        background-color: #D3D9D4;
        width: 20%;
        display: flex;
        flex-direction: column;
        justify-content: center;
        padding-left: 20px;
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
        background-color: #748D92;
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
    }

    .center-content {
        flex-grow: 1;
        overflow-y: auto;
    }

    .right {
        z-index: 1;
        background-color: #124E66;
        width: 40%;
        transform: translateX(-200%);
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-size: 24px;
        left: 60%;
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
</style>

<div class="container">
    <!-- Left column -->
    <div class="left">
        <h1>PROJECTS</h1>
        <div>
            <div class="subheadings" role="button" tabindex="0" 
                 on:click={() => changeContent(1)} 
                 on:keydown={(e) => e.key === 'Enter' && changeContent(1)}>SKINCANCER DETECTION</div>
            <div class="subheadings" role="button" tabindex="0" 
                 on:click={() => changeContent(2)} 
                 on:keydown={(e) => e.key === 'Enter' && changeContent(2)}>TABLEAU VISUALIZATION OF AIRBNB DATA</div>
            <div class="subheadings" role="button" tabindex="0" 
                 on:click={() => changeContent(3)} 
                 on:keydown={(e) => e.key === 'Enter' && changeContent(3)}>REINFORCEMENT LEARNING MODEL FOR SAME-DAY DELIVERY</div>
            <div class="subheadings" role="button" tabindex="0" 
                 on:click={() => changeContent(4)} 
                 on:keydown={(e) => e.key === 'Enter' && changeContent(4)}>PREDICTIVE ANALYTICS OF AIRLINE DELAYS</div>
        </div>
    </div>

    <!-- Center column -->
    <div class="center" 
         class:active={mounted && activeSubheading !== 0} 
         class:animate={animationTriggered}
         style="transform: translateX({centerTransform})">
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
        </div>
    </div>

    <!-- Right column -->
    <div class="right" 
         class:active={mounted && activeSubheading !== 0} 
         class:animate={animationTriggered}
         style="transform: translateX({rightTransform})">
        <h1>HEADING 3</h1>
    </div>
</div>
