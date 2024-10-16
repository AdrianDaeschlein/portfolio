<script lang="ts">
  import { onMount } from 'svelte';

  let highlightedCode = '';

  // Prism.js for syntax highlighting
  onMount(async () => {
    const Prism = await import('prismjs');
    await import('prismjs/themes/prism-okaidia.css');
    await import('prismjs/components/prism-python.js'); // For Python language
    highlightedCode = Prism.highlight(code, Prism.languages.python, 'python');
  });

  const code = `from typing import Callable, List, Tuple

from isek_analyze.indexing import functions as f
from isek_lib import now
from isek_lib.interfaces import CalculatorInterface
from isek_lib.types import (
    ACTIVATION_FUNCTION_TYPES,
    ACTIVATION_LIN_PARAMS,
    ACTIVATION_SIG_PARAMS,
    ACTIVATION_TANH_PARAMS,
    IndexingConfig,
    Task,
)

sys_type = type


class Calculator(CalculatorInterface):
    """Calculate the index of a task with different activation functions and variations"""

    def __init__(self, type: ACTIVATION_FUNCTION_TYPES, variation: int | tuple) -> None:
        """Init the Calculator with custom parameters

        Args:
            type (ACTIVATION_FUNCTION_TYPES): Type of the activation function.
            variation (int | Tuple): Index of the variation oder Tuple of custom parameters.
        """

        if sys_type(variation) == int:
            self.activation_function = self._get_function(type, variation)
        else:
            self.activation_function = self._get_function_with_custom_variation(type, variation)

    @classmethod
    def from_config(cls, config: IndexingConfig):
        """Init the Calculator from the config"""

        return cls(config.activation_function, config.activation_variation)

    def _get_variation(self, var: List[Tuple], i: int) -> tuple:
        """Get variation of the activation function params

        Args:
            var (List[Tuple]): List of params
            i (int): Index of the variation

        Returns:
            variation: params for the variation of the activation function
        """

        try:
            return var[i]
        except IndexError:
            return var[0]

    def _get_function(
        self, type: ACTIVATION_FUNCTION_TYPES, variation: int
    ) -> Callable[[int | float, int | float], float]:
        """Get activation function of the given type

        Args:
            type (ACTIVATION_FUNCTION_TYPES): Type of the activation function
            variation (int): Index of the variation

        Returns:
            activation_function: Activation function
        """

        if type is ACTIVATION_FUNCTION_TYPES.SIG:
            return lambda p, x: f.sig(p, x, *self._get_variation(ACTIVATION_SIG_PARAMS, variation))
        elif type is ACTIVATION_FUNCTION_TYPES.TANH:
            return lambda p, x: f.tanh(p, x, *self._get_variation(ACTIVATION_TANH_PARAMS, variation))
        elif type is ACTIVATION_FUNCTION_TYPES.LIN:
            return lambda p, x: f.lin_sum(p, x, *self._get_variation(ACTIVATION_LIN_PARAMS, variation))

    def _get_function_with_custom_variation(
        self, type: ACTIVATION_FUNCTION_TYPES, params: Tuple
    ) -> Callable[[int | float, int | float], float]:
        """Get activation function of the given type with a custom set of parameters

        Args:
            type (ACTIVATION_FUNCTION_TYPES): Type of the activation function
            params (Tuple): Custom Params

        Returns:
            activation_function: Activation function
        """

        if type is ACTIVATION_FUNCTION_TYPES.SIG:
            return lambda p, x: f.sig(p, x, *params)
        elif type is ACTIVATION_FUNCTION_TYPES.TANH:
            return lambda p, x: f.tanh(p, x, *params)
        elif type is ACTIVATION_FUNCTION_TYPES.LIN:
            return lambda p, x: f.lin_sum(p, x, *params)

    def run(self, task: Task):
        priority: int = 4 - task.priority
        deadline = task.deadline

        diff = deadline - now()
        diff_days: int = round(diff.total_seconds() / 3600 / 24)

        task.index = float(self.activation_function(priority, diff_days))
        return task

    def run_relative(self, task_a: Task, task_b: Task) -> float:
        """Calculate relative index between two tasks

        Args:
            task_a (Task): First Task
            task_b (Task): Second Task

        Returns:
            float: calculated index
        """

        priority: int = 4 - task_a.priority
        deadline_a = task_a.deadline
        deadline_b = task_b.deadline

        diff = deadline_a - deadline_b
        diff_days: int = round(diff.total_seconds() / 3600 / 24)

        index = self.activation_function(priority, diff_days)
        return index
  
      `;
</script>

<div class="code-container">
  <div class="header">
    <h3>Code</h3>
    <a
      href="https://gitlab.com/ise-tech-developments"
      target="_blank"
      rel="noopener noreferrer"
      title="Open in GitLab"
    >
      <img src="/gitlab-clipart.svg" alt="GitLab" class="gitlab-icon" />
    </a>
  </div>
  <div class="code-editor">
    <pre>{@html highlightedCode}</pre>
  </div>
</div>

<!-- <style>
    .code-container {
      display: flex;
      flex-direction: column;
      align-items: flex-start;
      width: 100%;
      height: 100%;
      margin: 40px;
      margin-left: 80px;
    }
  
    .code-editor {
      background-color: #2d2d2d;
      color: #f8f8f2;
      padding: 16px;
      border-radius: 8px;
      font-family: 'Fira Code', monospace;
      font-size: 0.9rem;
      overflow: auto;
      height: 80%;
      max-height: 800px;
      width: 100%;
      max-width: 400px;
    }
  
    pre {
      margin: 0;
    }
  </!-->
-->

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

  .gitlab-icon {
    width: 24px;
    height: 24px;
    margin-left: 10px;
  }

  .code-editor {
    background-color: #2d2d2d;
    color: #f8f8f2;
    padding: 16px;
    border-radius: 8px;
    font-family: 'Fira Code', monospace;
    font-size: 0.9rem;
    overflow: auto;
    height: 80%;
    max-height: 800px;
    width: 100%;
    max-width: 400px;
  }

  pre {
    margin: 0;
  }
</style>
