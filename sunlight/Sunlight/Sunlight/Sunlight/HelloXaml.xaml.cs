using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

using Xamarin.Forms;
using Xamarin.Forms.Xaml;

namespace Sunlight
{
    [XamlCompilation(XamlCompilationOptions.Compile)]
    public partial class HelloXaml : ContentPage
    {
        public HelloXaml()
        {
            InitializeComponent();
        }

        private async void OnButtonClicked(Object sender, EventArgs args)
        {
            Button button = (Button) sender;
            await DisplayAlert("Clicked!",
                "The button labeled '" + button.Text + "' has been clicked",
                "OK");
            await Navigation.PushAsync(new HelloGrid());

        }
    }
}