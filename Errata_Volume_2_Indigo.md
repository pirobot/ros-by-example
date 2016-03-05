# Errata for Volume 2 ROS Indigo #

The following errata are for book version **1.1.1**.

   * In Section **3.4: A Patrol Bot Example**, the bullet point that reads:

   **Sensors and actuators:** ros-indigo-fake-localization

   should simply be:

   **Sensors and actuators:**

   * In Section **3.10.2: Basic components of the pi_trees library** under the definition of a Sequence, the sentence:

   A Sequence runs each child task in list order until one **succeeds** or until it runs out of subtasks.

   should read:

   A Sequence runs each child task in list order until one **fails** or until it runs out of subtasks.

   * In Section **3.10.7 Adding and removing tasks**, inside the code block the task named COUNT\_WORDS should be COUNT\_TO\_10.  In other words, the lines that read:
```python
    if remove:
        PARALLEL_DEMO.remove_child(COUNT_WORDS)
    else:
        PARALLEL_DEMO.add_child(COUNT_WORDS)
```
   should be:
```python
    if remove:
        PARALLEL_DEMO.remove_child(COUNT_TO_10)
    else:
        PARALLEL_DEMO.add_child(COUNT_TO_10)
```

