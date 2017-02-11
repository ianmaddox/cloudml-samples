package com.google.cloud.ml.samples.textclassification;

import com.google.cloud.bigquery.BigQuery;
import com.google.cloud.bigquery.BigQueryOptions;
import com.google.cloud.bigquery.DatasetId;
import com.google.cloud.bigquery.DatasetInfo;
import com.google.cloud.bigquery.ExtractJobConfiguration;
import com.google.cloud.bigquery.Job;
import com.google.cloud.bigquery.JobInfo;
import com.google.cloud.bigquery.Table;
import com.google.cloud.bigquery.TableId;
import com.google.cloud.bigquery.TableInfo;
import com.google.cloud.bigquery.ViewDefinition;
import java.util.Collections;
import java.util.concurrent.TimeoutException;
import org.apache.spark.SparkConf;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

/**
 * Created by elibixby on 2/1/17.
 */
public class TextClassificationPreprocessor {

  static String QUERY = "SELECT subreddit, title FROM [fh-bigquery:reddit_posts.full_corpus_201512]" +
      "WHERE subreddit IN ( SELECT subreddit FROM (" +
      "SELECT subreddit, COUNT(subreddit) FROM [fh-bigquery:reddit_posts.full_corpus_201512]" +
      "GROUP BY subreddit ORDER BY 2 DESC LIMIT %i))";

  static String EXPORT_FILE_PATTERN = "/text_preprocessor/output*.avro";

  static String TEMP_DATASET = "temp_preprocessor";
  static String TEMP_TABLE = "temp_table";


  static void exportTable(String filePattern, int numClasses, String projectId)
      throws TimeoutException, InterruptedException {

    BigQuery bigquery = BigQueryOptions.getDefaultInstance().getService();
    DatasetId datasetId = DatasetId.of(projectId, TEMP_DATASET);
    com.google.cloud.bigquery.Dataset dataset = bigquery.getDataset(datasetId);

    if (dataset == null){
      dataset = bigquery.create(DatasetInfo.of(datasetId));
    }

    TableId tempTable = TableId.of(projectId, TEMP_DATASET, TEMP_TABLE);
    Table table = bigquery.create(TableInfo.of(
        tempTable,
        ViewDefinition.of(String.format(QUERY, numClasses))
    ));

    ExtractJobConfiguration jobConfig = ExtractJobConfiguration
        .newBuilder(tempTable, Collections.singletonList(filePattern))
        .setCompression("GZIP")
        .setFormat("AVRO")
        .build();

    Job extractJob = bigquery.create(JobInfo.of(jobConfig));

    extractJob.waitFor();
  }


  public static void main(String[] args) throws TimeoutException, InterruptedException {
    String basePath = args[0];
    int numClasses = Integer.parseInt(args[1]);
    SparkSession sp = SparkSession.builder().appName("TFTextClassifier").getOrCreate();
    SparkConf sc = sp.sparkContext().conf();
    String projectId = sc.get("fs.gs.project.id");
    String filePattern = sc.get("fs.default.name") + EXPORT_FILE_PATTERN;

    exportTable(filePattern, numClasses, projectId);
    
    Dataset<Row> df = sp.read().format("com.databricks.spark.avro")
        .load(filePattern)

  }

}
