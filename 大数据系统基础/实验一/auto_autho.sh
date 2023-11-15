#!/bin/bash

# check the exist of id_rsa.pub
if [ ! -f ~/.ssh/id_rsa.pub ]; then
    ssh-keygen -t rsa -b 4096 -N "" -f ~/.ssh/id_rsa
fi


# ssh-copy-id to three other nodes
for i in {3..5}
do
    echo "Copying key to node: $node..."
    sshpass -p2023214278 ssh-copy-id -i ~/.ssh/id_rsa.pub -o StrictHostKeyChecking=no "thumm0$i"
done

# print done message
echo "<--------Done-------->"

