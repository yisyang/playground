using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Xamarin.Forms;

namespace Sunlight
{
    public partial class MainPage : ContentPage
    {
        public MainPage()
        {
            InitializeComponent();
            
            Button buttonHello = new Button
            {
                Text = "Hello!",
                HorizontalOptions = LayoutOptions.Center,
                VerticalOptions = LayoutOptions.Center
            };

            buttonHello.Clicked += async (sender, args) =>
            {
                await Navigation.PushAsync(new HelloXaml());
            };
            
            Button buttonGrid = new Button
            {
                Text = "Grid!",
                HorizontalOptions = LayoutOptions.Center,
                VerticalOptions = LayoutOptions.Center
            };

            buttonGrid.Clicked += async (sender, args) =>
            {
                await Navigation.PushAsync(new HelloGrid());
            };

            Content = new StackLayout
            {
                Margin = new Thickness(40),
                Children =
                {
                    buttonHello,
                    buttonGrid
                }
            };
        }
    }
}
