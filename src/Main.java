import weka.core.Attribute;
import weka.core.Instances;
import weka.core.converters.CSVLoader;
import weka.core.converters.CSVSaver;

import java.io.File;
import java.util.ArrayList;
import java.util.List;

public class Main {
    public static void main(String[] args) throws Exception {
        // Load CSV file
        CSVLoader loader = new CSVLoader();
        loader.setSource(new File("C:\\Users\\DELL\\Downloads\\cleaned_data.csv")); // Specify the CSV file path
        Instances dataset = loader.getDataSet(); // Load the CSV as Instances

        // Print dataset summary before dropping columns
        System.out.println("Original Dataset Summary:");
        System.out.println(dataset.toSummaryString());

        // List of columns to drop
        String[] columnsToDrop = {
                "homepage", "tagline", "backdrop_path", "production_companies",
                "production_countries", "spoken_languages", "poster_path", "overview"
        };

        // Get attribute indices to drop
        List<Integer> indicesToDrop = new ArrayList<>();
        for (String columnName : columnsToDrop) {
            Attribute attribute = dataset.attribute(columnName);
            if (attribute != null) {
                indicesToDrop.add(attribute.index());
            }
        }

        // Sort indices in descending order to safely remove attributes
        indicesToDrop.sort((a, b) -> b - a);

        // Drop columns
        for (int index : indicesToDrop) {
            dataset.deleteAttributeAt(index);
        }

        // Set "vote_average" as the class (target) attribute
        Attribute targetAttribute = dataset.attribute("vote_average");
        if (targetAttribute != null) {
            dataset.setClass(targetAttribute);
        } else {
            System.err.println("Error: 'vote_average' column not found!");
            return;
        }

        // Print dataset summary after dropping columns
        System.out.println("Updated Dataset Summary:");
        System.out.println(dataset.toSummaryString());

        // Save the updated dataset as a new CSV file
        CSVSaver saver = new CSVSaver();
        saver.setInstances(dataset);
        saver.setFile(new File("C:\\Users\\DELL\\Downloads\\output.csv")); // Specify the output CSV file path
        saver.writeBatch(); // Write the dataset to the new file
    }
}
