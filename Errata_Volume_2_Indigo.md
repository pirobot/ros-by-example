# Errata for Volume 2 ROS Indigo #

The following errata are for book version **1.1.1**.

  * In Section **3.10.2 Basic components of the pi_trees library** under the definition of a Sequence, the sentence:

"A Sequence runs each child task in list order until one **succeeds** or until it runs out of
subtasks."

should read:

"A Sequence runs each child task in list order until one **fails** or until it runs out of
subtasks."
 
