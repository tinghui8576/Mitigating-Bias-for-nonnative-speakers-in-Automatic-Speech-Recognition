import requests
import pandas as pd
from bs4 import BeautifulSoup
import time
import sys
import re
import os
ROOT_URL = 'http://accent.gmu.edu/'
BROWSE_LANGUAGE_URL = 'browse_language.php?function=find&language={}'
LANGUAGE_LIST_URL = 'browse_language.php'  # URL for the page listing languages
WAIT = 1.2
DEBUG = True

def get_htmls(urls):
    '''
    Retrieves html in text form from ROOT_URL
    :param urls (list): List of urls from which to retrieve html
    :return (list): list of HTML strings
    '''
    htmls = []
    for url in urls:
        if DEBUG:
            print('Downloading from {}'.format(url))
        htmls.append(requests.get(url).text)
        time.sleep(WAIT)

    return(htmls)

def get_available_languages():
    '''
    Scrapes the available languages from the website
    :return (list): List of available languages
    '''
    response = requests.get(ROOT_URL + LANGUAGE_LIST_URL)
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Scraping the language names from the website
    language_links = []
    for link in soup.find_all('a', href=True):
        if 'browse_language.php?function=find&language=' in link['href']:
            language = link.text.strip()
            if language not in language_links:  # Avoid duplicates
                language_links.append(language)

    return language_links

def build_search_urls(languages):
    '''
    creates url from ROOT_URL and languages
    :param languages (list): List of languages
    :return (list): List of urls
    '''
    return([ROOT_URL+BROWSE_LANGUAGE_URL.format(language) for language in languages])

def parse_p(p_tag):
    '''
    Extracts href property from HTML <p> tag string
    :param p_tag (str): HTML string
    :return (str): string of link
    '''
    text = p_tag.text.replace(' ','').split(',')
    return([ROOT_URL+p_tag.a['href'], text[0], text[1]])

def get_bio(hrefs):
    '''
    Retrieves HTML from list of hrefs and returns bio information
    :param hrefs (list): list of hrefs
    :return (DataFrame): Pandas DataFrame with bio information
    '''

    htmls = get_htmls(hrefs)
    bss = [BeautifulSoup(html,'html.parser') for html in htmls]
    rows = []
    bio_row = []
    for bs in bss:
        rows.append([li.text for li in bs.find('ul','bio').find_all('li')])
    for row in rows:
        bio_row.append(parse_bio(row))

    return(pd.DataFrame(bio_row))

def parse_bio(row):
    '''
    Parse bio data from row string
    :param row (str): Unparsed bio string
    :return (list): Bio columns
    '''
    cols = []
    for col in row:
        try:
            tmp_col = re.search((r"\:(.+)",col.replace(' ','')).group(1))
        except:
            tmp_col = col
        cols.append(tmp_col)
    return(cols)


def create_dataframe(languages):
    '''

    :param languages (str): language from which you want to get html
    :return df (DataFrame): DataFrame that contains all audio metadata from searched language
    '''
    htmls = get_htmls(build_search_urls(languages))
    bss = [BeautifulSoup(html,'html.parser') for html in htmls]
    persons = []

    for bs in bss:
        for p in bs.find_all('p'):
            if p.a:
                persons.append(parse_p(p))

    df = pd.DataFrame(persons, columns=['href','language_num','sex'])

    bio_rows = get_bio(df['href'])

    if DEBUG:
        print('loading finished')

    df['birth_place'] = bio_rows.iloc[:,0]
    df['native_language'] = bio_rows.iloc[:,1]
    df['other_languages'] = bio_rows.iloc[:,2]
    df['age_sex'] = bio_rows.iloc[:,3]
    df['age_of_english_onset'] = bio_rows.iloc[:,4]
    df['english_learning_method'] = bio_rows.iloc[:,5]
    df['english_residence'] = bio_rows.iloc[:,6]
    df['length_of_english_residence'] = bio_rows.iloc[:,7]

    df['birth_place'] = df['birth_place'].apply(lambda x: x[:-6].split(' ')[-2:])
    # print(df['birth_place'])
    # df['birth_place'] = lambda x: x[:-6].split(' ')[2:], df['birth_place']
    df['native_language'] = df['native_language'].apply(lambda x: x.split(' ')[2])
    # print(df['native_language'])
    # df['native_language'] = lambda x: x.split(' ')[2], df['native_language']
    df['other_languages'] = df['other_languages'].apply(lambda x: x.split(' ')[2:])
    # print(df['other_languages'])
    # df['other_languages'] = lambda x: x.split(' ')[2:], df['other_languages']
    df['age_sex'], df['age'] = df['age_sex'].apply(lambda x: x.split(' ')[2:]), df['age_sex'].apply(lambda x: x.replace('sex:','').split(',')[1])
    # print(df['age'])
    # df['age_sex'] = lambda x: x.split(' ')[2], df['age_sex']
    # df['age_of_english_onset'] = lambda x: float(x.split(' ')[-1]), df['age_of_english_onset']
    df['age_of_english_onset'] = df['age_of_english_onset'].apply(lambda x: float(x.split(' ')[-1]))
    # print(df['age_of_english_onset'])
    # df['english_learning_method'] = lambda x: x.split(' ')[-1], df['english_learning_method']
    df['english_learning_method'] = df['english_learning_method'].apply(lambda x: x.split(' ')[-1])
    # print(df['english_learning_method'])
    # df['english_residence'] = lambda x: x.split(' ')[2:], df['english_residence']
    df['english_residence'] = df['english_residence'].apply(lambda x: x.split(' ')[2:])
    # print(df['english_residence'])
    # df['length_of_english_residence'] = lambda x: float(x.split(' ')[-2]), df['length_of_english_residence']
    df['length_of_english_residence'] = df['length_of_english_residence'].apply(lambda x: float(x.split(' ')[-2]))
    # print(df['length_of_english_residence'])

    # df['age'] = lambda x: x.replace(' ','').split(',')[0], df['age_sex']
    return df
def save_dataframe_to_csv(df, destination_file):
    '''
    Save the DataFrame to CSV, appending if the file exists, and writing headers only once.
    :param df: DataFrame to save
    :param destination_file: CSV file path
    '''
    # If the file exists, append the DataFrame, otherwise create the file
    if os.path.exists(destination_file):
        df.to_csv(destination_file, mode='a', header=False, index=False)
    else:
        df.to_csv(destination_file, mode='w', header=True, index=False)
def remove_duplicates(input_file, output_file):
    # Load the CSV file into a DataFrame
    df = pd.read_csv(input_file)
    
    # Remove duplicates across all columns
    df_cleaned = df.drop_duplicates()

    # Optionally, if you want to remove duplicates based on certain columns, you can specify those columns:
    # df_cleaned = df.drop_duplicates(subset=['column_name1', 'column_name2'])

    # Save the cleaned DataFrame back to a new CSV file
    df_cleaned.to_csv(output_file, index=False)
    print(f"Duplicates removed and saved to {output_file}")

if __name__ == '__main__':
    '''
    console command example:
    python fromwebsite.py bio_metadata.csv
    '''

    df = None

    # Set destination file
    destination_file = sys.argv[1]

    # Get all available languages from the website
    languages = get_available_languages()
    print(f"Found {len(languages)} languages.")

    # Initialize an empty DataFrame to accumulate the data
    df = pd.DataFrame()

    # Loop through each language and process its data
    for language in languages:
        try:
            print(f"Processing language: {language}")
            # Create dataframe for the current language
            language_df = create_dataframe([language])
            
            if not language_df.empty:
                # Append to the main DataFrame only if the language data is not empty
                #df = df.append(language_df, ignore_index=True)

                # Save the current data to the CSV (appending if it already exists)
                save_dataframe_to_csv(language_df, destination_file)
                print(f"Saved CSV after processing {language}")
            else:
                print(f"No data to save for language: {language}")

        except Exception as e:
            print(f"Error processing {language}: {e}")



    remove_duplicates(destination_file, destination_file)

    print("Processing complete!")