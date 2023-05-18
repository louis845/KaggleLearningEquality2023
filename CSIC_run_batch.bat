REM The following command is for converting the csv files to text embeddings. Uncomment the line (REM) to run it.
REM python full_text_to_vectors_pipeline.py

REM Unsupervised dimension reduction methods on the text embeddings
python CSIC_unsupervised_conversion.py --dimension_reduction_method PCA --target_dimensions 192
python CSIC_unsupervised_conversion.py --dimension_reduction_method PCA --target_dimensions 96
python CSIC_unsupervised_conversion.py --dimension_reduction_method PCA --target_dimensions 48
python CSIC_unsupervised_conversion.py --dimension_reduction_method isomap --target_dimensions 192
python CSIC_unsupervised_conversion.py --dimension_reduction_method isomap --target_dimensions 96
python CSIC_unsupervised_conversion.py --dimension_reduction_method isomap --target_dimensions 48

REM Combine with the graph structure
python CSIC_neighbors_text_embeddings.py