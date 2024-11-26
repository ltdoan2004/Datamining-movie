# Weka API Project Setup

This guide will walk you through setting up the Weka API in your Java project using IntelliJ IDEA, configuring JVM options to allocate 4GB of heap memory, and resolving access issues by adding the required JVM argument.

---

## Prerequisites

Before proceeding, ensure you have the following installed:
- IntelliJ IDEA (Community or Ultimate Edition)
- JDK (Java Development Kit) 8
- Weka library JAR file (or Maven dependency)

---

## Steps to Set Up Weka API

### 1. **Download the Weka Library**
   - Visit the [Weka official website](https://www.cs.waikato.ac.nz/ml/weka/) to download the `weka.jar` file.
   - Alternatively, if using Maven, add the following dependency to your `pom.xml`:
     ```xml
     <dependency>
         <groupId>nz.ac.waikato.cms.weka</groupId>
         <artifactId>weka-stable</artifactId>
         <version>3.8.6</version> <!-- Replace with the latest version -->
     </dependency>
     ```

### 2. **Add Weka to Your IntelliJ Project**
   - If using the JAR file:
     1. Create a `lib` folder in your project directory.
     2. Place the downloaded `weka.jar` file inside the `lib` folder.
     3. Add the JAR file to the project:
        - Right-click on your project > `Open Module Settings` > `Libraries` > `+` > Select `weka.jar` from the `lib` folder.

### 3. **Install the MTJ (Matrix Toolkit for Java) Library**
   - Weka depends on the MTJ library for matrix operations. If you're not using Maven, download the MTJ JAR file and add it to your project's `lib` folder, just like the Weka JAR.

   - If using Maven, add the MTJ dependency:
     ```xml
     <dependency>
         <groupId>net.sf.s2</groupId>
         <artifactId>mtj</artifactId>
         <version>0.9.14</version>
     </dependency>
     ```

---

## Configuring IntelliJ IDEA

### 1. **Increase JVM Heap Memory to 4GB**
   - Follow these steps to allocate more memory:
     1. Go to `Run > Edit Configurations`.
     2. Select your application's run configuration.
     3. In the **VM Options** field, add:
        ```
        -Xmx4G
        ```

### 2. **Fix Module Access Issues**
   - To avoid the `java.lang.reflect.InaccessibleObjectException`, add the following JVM argument:
     ```
     --add-opens java.base/java.lang=ALL-UNNAMED
     ```

     Combine this with the heap size argument in the **VM Options** field:
     ```
     -Xmx4G --add-opens java.base/java.lang=ALL-UNNAMED
     ```

     Example:
     - **Final VM Options**:
       ```
       -Xmx4G --add-opens java.base/java.lang=ALL-UNNAMED
       ```

---

## Sample Code

Here is an example program to test the Weka API setup:

```java
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class WekaSetupTest {
    public static void main(String[] args) throws Exception {
        DataSource source = new DataSource("path/to/dataset.arff"); // Replace with your dataset file path
        Instances data = source.getDataSet();
        System.out.println("Dataset loaded: " + data.numInstances() + " instances");
    }
}
