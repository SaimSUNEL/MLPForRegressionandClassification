#ifndef DATALOAD_H_INCLUDED
#define DATALOAD_H_INCLUDED

#define LOAD_DATA_SIZE 4
Vector2D * load_text_data()
{

    FILE * dosya = fopen("text_data.dat", "rb");

    int width, height;
    fread(&width, sizeof(int), 1, dosya);
    fread(&height, sizeof(int), 1, dosya);
     float * loaded_data = malloc(width*height*sizeof(float));
    for(int a =0; a< width*height; a++)
        fread(&loaded_data[a], sizeof(float), 1, dosya);


    fclose(dosya);

    printf("Width : %d - Height : %d\n", width, height);

   Vector2D * vec = CreateVector2D(loaded_data, height, width);


  return vec;


}

Vector2D * load_label_data()
{

    FILE * dosya = fopen("label_data.dat", "rb");

    int width, height;
    fread(&width, sizeof(int), 1, dosya);
    fread(&height, sizeof(int), 1, dosya);
     float * loaded_data = (float *)malloc(width*height*sizeof(float));
    int value;
    for(int a =0; a< width*height; a++)
        {

        fread(&value, sizeof(int), 1, dosya);
		loaded_data[a] = value;
		}

    fclose(dosya);

    printf("Width : %d - Height : %d\n", width, height);

   Vector2D * vec = CreateVector2D(loaded_data, height, width);


  return vec;


}


Vector2D * load_test_text_data()
{

    FILE * dosya = fopen("test_text_data.dat", "rb");

    int width, height;
    fread(&width, sizeof(int), 1, dosya);
    fread(&height, sizeof(int), 1, dosya);
     float * loaded_data = malloc(width*height*sizeof(float));
    for(int a =0; a< width*height; a++)
        fread(&loaded_data[a], sizeof(float), 1, dosya);


    fclose(dosya);

    printf("Width : %d - Height : %d\n", width, height);

   Vector2D * vec = CreateVector2D(loaded_data, height, width);


  return vec;


}


Vector2D * load_test_label_data()
{

    FILE * dosya = fopen("test_label_data.dat", "rb");

    int width, height;
    fread(&width, sizeof(int), 1, dosya);
    fread(&height, sizeof(int), 1, dosya);
     float * loaded_data = (float *)malloc(width*height*sizeof(float));
    int value;
    for(int a =0; a< width*height; a++)
        {

        fread(&value, sizeof(int), 1, dosya);
		loaded_data[a] = value;
		}

    fclose(dosya);

    printf("Width : %d - Height : %d\n", width, height);

   Vector2D * vec = CreateVector2D(loaded_data, height, width);


  return vec;


}




#endif // DATALOAD_H_INCLUDED
