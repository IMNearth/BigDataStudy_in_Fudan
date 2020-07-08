#!/usr/bin/env bash
rm ./part-00000
hdfs dfs -rm -r bigdata/output/


# sprcify -D <options> at the beginning
# otherwise hadoopstreaming will fail to recognize there options
hadoop jar $HADOOP_HOME/share/hadoop/tools/lib/hadoop-streaming-2.9.2.jar \
-D map.output.key.field.separator=@ \
-D mapreduce.partition.keypartitioner.options=-k1,1 \
-D mapreduce.job.output.key.comparator.class=org.apache.hadoop.mapreduce.lib.partition.KeyFieldBasedComparator \
-D mapreduce.partition.keycomparator.options=-k2nr \
-input bigdata/inputs/news_tensite_xml.smarty.dat \
-output bigdata/output/ \
-mapper "python3 mapper.py" \
-reducer "python3 reducer.py" \
-partitioner org.apache.hadoop.mapred.lib.KeyFieldBasedPartitioner \


# get file from HDFS to local and visualize them
hdfs dfs -get bigdata/output/part-00000 ./
head -100 part-00000


# Tips:
# if we do not use "\t" to seperate key/value pair
# we need to specify
# -D stream.map.output.field.separator=@ \
# -D stream.num.map.output.key.fields=2 \