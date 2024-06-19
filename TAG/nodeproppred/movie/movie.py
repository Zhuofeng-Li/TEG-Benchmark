import json
import os.path as osp
from typing import List
import gzip
import shutil
import torch
from torch_geometric.data import InMemoryDataset, HeteroData


class Amazon_Movies(InMemoryDataset):  
    def __init__(self, root: str) -> None:
        super().__init__(root)
        self.load(self.processed_paths[0], data_cls=HeteroData)

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root,'movie_dataset','raw') 

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, 'movie_dataset', 'processed')

    @property
    def raw_file_names(self) -> List[str]:
        file_names = [
            'node-feat', 'node-label', 'relations', 'split',
            'num-node-dict.csv.gz'
        ]

        return file_names

    @property
    def processed_file_names(self) -> str:
        return 'data.pt'

    def process(self) -> None:
        path = osp.join(self.raw_dir, 'reviews_Movies_and_TV_5.json')
        genre_path = osp.join(self.raw_dir, 'meta_Movies_and_TV.json')

        final_data = []
        categories = ['Movies & TV', 'Movies', 'Studio Specials', 'Warner Home Video', 'All Titles', 'Science Fiction & Fantasy', 'Science Fiction', 'Animation', 'Sony Pictures Home Entertainment', 'All Sony Pictures Titles', 'Genre for Featured Categories', 'Music Videos & Concerts', 'Widescreen', 'Comedy', 'Yoga', 'General', 'Action & Adventure', 'A&E Home Video', 'All A&E Titles', 'Musicals & Performing Arts', 'Ballet & Dance', 'Sony Pictures Classics', 'All Sony Pictures Classics', '20th Century Fox Home Entertainment', 'Musicals', 'Futuristic', 'Boxed Sets', 'Cult Movies', 'Drama', 'Criterion Collection', 'All', 'Art House & International', 'By Country', 'Sweden', 'Fantasy', 'Japan', 'Classics', 'PBS', 'United Kingdom', 'Documentary', 'Warner Video Bargains', 'Westerns', 'Horror', 'Hong Kong', 'Lionsgate Home Entertainment', 'All Lionsgate Titles', 'John Wayne Store', 'Lionsgate DVDs Under $10', 'Paramount Home Entertainment', 'HBO', 'All HBO Titles', 'Kids & Family', 'Universal Studios Home Entertainment', 'All Universal Studios Titles', 'Mystery & Thrillers', 'Alien Invasion', 'DreamWorks', 'Top Sellers', 'Robots & Androids', 'Lionsgate DVDs Under $15', 'Disney Home Video', 'By Age', '3-6 Years', 'Miramax Home Entertainment', 'Oscar Collection', 'Walt Disney Studios Home Entertainment', 'Animated Movies', 'All Disney Titles', 'TV', 'Characters & Series', 'Mary-Kate & Ashley', 'Sci-Fi Action', 'Aliens', 'Music Artists', 'Presley, Elvis', 'Manilow, Barry', 'MGM Home Entertainment', 'All MGM Titles', 'All Fox Titles', 'Action', 'Germany', 'MGM Movie Time', 'MGM DVDs Under $15', 'Music & Musicals', 'Shakespeare on DVD Store', 'The Works', 'The Histories', 'Sci Fi Channel', 'All Sci Fi Channel Shows', 'Star Wars', 'Holidays & Seasonal', 'Christmas', 'Independently Distributed', 'Special Interests', 'Military & War', 'Cirque du Soleil', 'Exercise & Fitness', 'Faith & Spirituality', 'Sci-Fi & Fantasy', 'Feature Films', 'Brazil', 'By Genre', 'LGBT', 'Spain', 'New Yorker Films', 'All New Yorker Titles', 'France', 'Mystery & Suspense', 'Classical', 'Animated Cartoons', 'Italy', 'Jewish Heritage', 'Holocaust', 'Wellspring Home Video', 'American Masters Collection', 'By Original Language', 'French', 'Television', 'NOVA', 'Anime & Manga', 'Cash, Johnny', 'The Tragedies', 'Romance', 'Silent Films', 'Yes', 'Family Features', 'German', 'Sports', 'KISS', 'Live Action', 'Russian', 'BBC', 'All BBC Titles', 'Haggard, Merle', 'Foreign Films', 'India', 'Fox TV', 'Russia', 'By Instructor', 'Patricia Walden', 'Australia & New Zealand', 'Weird Al', 'Opera', 'Jackson, Michael', 'Blaxploitation', 'Italian', 'The Beatles', 'Ringo Starr', 'Africa', 'U2', 'Mexico', 'Hendrix, Jimi', 'Sci-Fi Series & Sequels', 'Godzilla', 'China', '7-11 Years', 'Japanese', 'Hungarian', 'Canada', 'Indie & Art House', 'Poland', 'African American Cinema', 'TV & Miniseries', 'Other Instructors', 'Turner, Tina', 'Spanish', 'Fitness', 'Ireland', 'Taiwan', 'Aerosmith', 'Nelson, Willie', 'Art & Artists', 'Miramax Home Video', 'Animated Characters', 'The Comedies', 'Norwegian', 'Cuba', 'By Animator', 'Mamoru Oshii', 'Argentina', 'Camp', 'Rodney Yee', 'Grateful Dead', 'MGM Screen Epics', 'Korea', 'Swedish', 'French New Wave', 'Stop-Motion & Clay Animation', 'Chinese', 'World Class Cinema', 'MGM Vintage Classics', 'Full Moon Video', 'Christian Video', 'Bible', 'Iran', 'For the Whole Family', 'Health', 'Cops & Triads', 'Iceland', 'Biography', 'Mother Teresa', 'Shakur, Tupac', 'Carpenters', 'Yoga Studios', 'Yoga Journal', 'Bon Jovi', 'Anime', 'Spanish Language', 'Misterio y suspenso', 'Ciencia ficcin y fantasa', 'Fully Loaded DVDs', 'Special Editions', 'Genesis', 'Monsters & Mutants', 'Fitness & Yoga', 'Yoga Zone', 'Focus Features', 'Hong Kong Action', 'DTS', 'Armstrong, Louis', 'King, B.B.', 'John Lennon', '4-for-3 DVD', 'Wall-E', 'Religion & Spirituality', 'Adventures', 'Lionsgate DVDs Under $20', 'Noah', 'Bowie, David', 'Music Video & Concerts', 'Scooby Doo', 'Dr. Dre', 'James Bond', 'Collections & Documentaries', 'Exploitation', 'Suzanne Deason', 'Beastie Boys', 'Made-for-TV Movies', 'All Made-for-TV Movies', 'New Line Platinum Series', 'Queen', '7-9 Years', 'Broadway', 'Broadway Theatre Archive', 'Arabic', 'Educational', 'A&E Original Movies', 'Foreign Spotlight', 'Showtime', 'All Showtime Titles', 'Paul McCartney', 'Metheny, Pat', 'Prison', 'Rock, Chris', 'First to Know', 'Infinifilm Edition', 'Harry Potter', 'Harry Potter and the Prisoner of Azkaban', 'Reality TV', 'Harry Potter and the Order of the Phoenix', 'Marley, Bob', 'MTV', 'All MTV', 'Space Adventure', 'Docurama', 'Godsmack', 'Bee Gees', 'More to Explore', 'Modern Adaptations', 'Iron Maiden', 'History', 'Scooby Doo Animated Movies', 'Sting', 'Sundance Channel Home Entertainment', 'All Sundance Titles', 'Scooby Doo Live Action Movies', 'Clapton, Eric', 'John, Elton', 'Two-Disc Special Editions', 'African American Heritage', 'Film History & Film Making', 'Middle East', 'Pre & Post Natal', 'Classic TV', 'Jackass', 'Classics Kids Love', 'Pink Floyd', 'FX', 'All FX Shows', 'Estefan, Gloria', 'Madonna', 'The Rolling Stones', 'Aguilera, Christina', 'Ultimate Editions', 'Rich, Buddy', 'Christian Movies & TV', 'Dance & Music', 'Davis, Miles', 'The Comedy Central Store', 'Comedy Central Roast', 'Birth-2 Years', 'Rieu, Andre', 'Romantic', 'Subversive Cinema', 'Box Sets', 'Lifetime Original Movies', 'Other', 'Stand Up', 'The Doors', 'ABBA', '10-12 Years', 'Rush', 'The Twilight Zone', 'Twilight Zone DVDs', 'Disney Channel', 'Disney Channel Original Movies', 'Discovery Channel', 'Channels', 'Basie, Count', 'Dylan, Bob', 'Snoop Dogg', 'Blakey, Art', 'Shakespeare 101', 'Historical Context', 'Waters, Muddy', 'TV Talk Shows', 'Playing Shakespeare', 'Acting Troupes & Companies', 'Cooper, Alice', 'Performing Arts', 'Music & Performing Arts', 'Religion', 'HD DVD', 'Politics', 'Deep Purple', 'Mini-DVD', 'MOD CreateSpace Video', 'Walt Disney Legacy Collection', 'Santana', 'Walt Disney Treasures', 'Fleetwood Mac', 'Brooks, Garth', 'Animal Planet', 'Celtic Woman', 'History Channel', 'The History Channel Presents', 'Osbourne, Ozzy', 'Travel Channel', 'Passport to Europe', 'Queensryche', 'Cartoon Network', 'British Television', 'Blu-ray', 'Holiday, Billie', 'Shrek', 'Timeless Holiday Favorites', 'Comedy Central Presents', 'Harry Potter and the Half-Blood Prince', 'Terminator', 'All Terminator', 'Riverdance', 'Nature & Wildlife', 'Miniseries', 'Fox News', "Ultimate Collector's Editions", 'Hindi', 'TV News Programming', 'Disney Channel Series', 'Lionsgate Indie Selects', 'Crime & Conspiracy', 'Essential Art House', 'The Shins', 'Extended Editions', 'Roxy Music', 'Borge, Victor', 'X-Men', 'Motley Crue', 'Harry Potter and the Deathly Hallows', 'Danish', 'Carey, Mariah', 'Jane Austen on DVD Store', 'ABC TV Shows', 'Vaughan, Stevie Ray', 'Crow, Sheryl', 'CBS News Network', '60 Minutes Store', 'Emerson, Lake & Palmer', 'The Temptations', 'Young, Neil', 'Collins, Phil', 'Inspirational', 'Jesus', 'Hallmark Home Video', 'Dutch', 'Czech', 'Springsteen, Bruce', 'Vietnamese','Molly Fox','None']

        movieid2genre = {}
        print(path)
        with open(path) as f:
            for line in f:
                data = json.loads(line)
                final_data.append(data)

        with open(genre_path) as f:
            for line in f:
                data = json.loads(line)
                category_list = [1 if category in data['category'] else 0 for category in categories]
                movieid2genre[data['asin']] = category_list

        reviewer_id2idx = {}
        movie_id2idx = {}

        edge_index_reviewer_movie = []
        edge_label = []

        multi_label = []
        for item in final_data:
            reviewer_id = item['reviewerID']
            movie_id = item['asin']

            # reviewer-movie
            if reviewer_id not in reviewer_id2idx:
                reviewer_id2idx[reviewer_id] = len(reviewer_id2idx)
            if movie_id not in movie_id2idx:
                movie_id2idx[movie_id] = len(movie_id2idx)

            # 如果 movie_id 不在 movieid2genre 中，跳过此条记录
            if movie_id not in movieid2genre:
                continue

            # reviewer-movie edge
            edge_index_reviewer_movie.append([reviewer_id2idx[reviewer_id], movie_id2idx[movie_id]])
            # movie label
            multi_label.append(movieid2genre[movie_id])
            # edge label (rating)
            edge_label.append(item['overall'])

        # load to hetordata
        num_users = len(reviewer_id2idx)
        num_books = len(movie_id2idx)

        data = HeteroData()
        data['user'].x = torch.nn.init.xavier_uniform_(torch.Tensor(num_users, 1))  # TODO
        data['book'].x = torch.nn.init.xavier_uniform_(torch.Tensor(num_books, 1))
        data['book'].y = torch.tensor(multi_label).float()
        data['user', 'review', 'book'].edge_index = torch.tensor(edge_index_reviewer_movie,
                                                                      dtype=torch.long).t().contiguous()

        data['user', 'review', 'book'].edge_label = torch.tensor(edge_label)

        torch.manual_seed(66)
        torch.cuda.manual_seed(66)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # data split
        train_ratio = 0.8
        val_ratio = 0.1

        num_movie = data['book'].num_nodes
        num_train_movie = int(num_movie * train_ratio)
        num_val_movie = int(num_movie * val_ratio)
        num_test_movie = num_movie - num_train_movie - num_val_movie

        movie_indices = torch.randperm(num_movie)

        data['book'].train_mask = torch.zeros(num_movie, dtype=torch.bool)
        data['book'].val_mask = torch.zeros(num_movie, dtype=torch.bool)
        data['book'].test_mask = torch.zeros(num_movie, dtype=torch.bool)

        data['book'].train_mask[movie_indices[:num_train_movie]] = 1
        data['book'].val_mask[movie_indices[num_train_movie:num_train_movie + num_val_movie]] = 1
        data['book'].test_mask[movie_indices[-num_test_movie:]] = 1

        data.num_classes = 399

        self.save([data], self.processed_paths[0])

if __name__ == '__main__':
    Amazon_Movies(root='.')